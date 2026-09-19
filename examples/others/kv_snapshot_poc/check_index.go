// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Run from an llm-d-kv-cache checkout: go run /path/to/check_index.go cases.json
package main

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"sort"
	"strings"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

type testCase struct {
	FullCount int        `json:"full_count"`
	TailStart int        `json:"tail_start"`
	Name      string     `json:"name"`
	Full      [][]byte   `json:"full"`
	Snapshot  [][]byte   `json:"snapshot"`
	Suffix    [][]byte   `json:"suffix"`
	Tokens    [][]uint32 `json:"tokens"`
	BlockSize int        `json:"block_size"`
	Expected  *int       `json:"expected"`
}

type residency struct {
	medium string
	group  int
	hash   uint64
}

// The engine's AllBlocksCleared clears GPU only. Expand it to scoped removes
// before handing events to the existing Pool, whose Clear currently wipes CPU
// too. Everything else uses the unmodified vLLM adapter, Pool and index.
type adapter struct {
	kvevents.EngineAdapter
	refs map[residency]int
}

func (a *adapter) ParseMessage(m *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	pod, model, batch, err := a.EngineAdapter.ParseMessage(m)
	if err != nil {
		return pod, model, batch, err
	}
	out := make([]kvevents.GenericEvent, 0, len(batch.Events))
	for _, raw := range batch.Events {
		switch e := raw.(type) {
		case *kvevents.BlockStoredEvent:
			g := -1
			if e.GroupIdx != nil {
				g = *e.GroupIdx
			}
			for _, h := range e.BlockHashes {
				a.refs[residency{e.DeviceTier, g, h}]++
			}
			out = append(out, e)
		case *kvevents.BlockRemovedEvent:
			g := -1
			if e.GroupIdx != nil {
				g = *e.GroupIdx
			}
			for _, h := range e.BlockHashes {
				k := residency{e.DeviceTier, g, h}
				if a.refs[k] > 1 {
					a.refs[k]--
				} else {
					delete(a.refs, k)
				}
			}
			out = append(out, e)
		case *kvevents.AllBlocksClearedEvent:
			for k, n := range a.refs {
				if k.medium != "GPU" && k.medium != "" {
					continue
				}
				hashes := make([]uint64, n)
				for i := range hashes {
					hashes[i] = k.hash
				}
				var group *int
				if k.group != -1 {
					g := k.group
					group = &g
				}
				out = append(out, &kvevents.BlockRemovedEvent{BlockHashes: hashes, DeviceTier: k.medium, GroupIdx: group})
				delete(a.refs, k)
			}
		}
	}
	batch.Events = out
	return pod, model, batch, nil
}

func run(c testCase, payloads [][]byte) map[string][]string {
	ctx := context.Background()
	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	must(err)
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: c.BlockSize, HashSeed: "test"})
	must(err)
	a := &adapter{EngineAdapter: engineadapter.NewVLLMAdapter(), refs: make(map[residency]int)}
	cfg := kvevents.DefaultConfig()
	cfg.Concurrency = 1
	pool := kvevents.NewPool(cfg, idx, tp, a)
	for _, payload := range payloads {
		// Validate decoding synchronously: Pool logs parse errors rather than returning them.
		_, _, _, err := a.EngineAdapter.ParseMessage(&kvevents.RawMessage{Topic: "kv@pod@test-model", Payload: payload})
		must(err)
		pool.AddTask(&kvevents.RawMessage{Topic: "kv@pod@test-model", Payload: payload})
	}
	pool.Start(ctx)
	pool.Shutdown(ctx)
	result := make(map[string][]string)
	for _, tokens := range c.Tokens {
		keys, err := tp.TokensToKVBlockKeys(0, tokens, "test-model", nil)
		must(err)
		for _, key := range keys {
			found, err := idx.Lookup(ctx, []kvblock.BlockHash{key}, nil)
			must(err)
			for _, entry := range found[key] {
				result[key.String()] = append(result[key.String()], fmt.Sprint(entry))
			}
			sort.Strings(result[key.String()])
		}
	}
	return result
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
func main() {
	f, err := os.Open(os.Args[1])
	must(err)
	defer f.Close()
	var reader io.Reader = f
	if strings.HasSuffix(os.Args[1], ".gz") {
		compressed, err := gzip.NewReader(f)
		must(err)
		defer compressed.Close()
		reader = compressed
	}
	buffered := bufio.NewReader(reader)
	first, err := buffered.Peek(1)
	must(err)
	decoder := json.NewDecoder(buffered)
	var cases []testCase
	var shared struct {
		History [][]byte   `json:"history"`
		Cases   []testCase `json:"cases"`
	}
	if first[0] == '{' {
		must(decoder.Decode(&shared))
		cases = shared.Cases
		for i := range cases {
			c := &cases[i]
			c.Full = shared.History[:c.FullCount]
			c.Snapshot = append(c.Snapshot, shared.History[c.TailStart:c.FullCount]...)
		}
	} else {
		must(decoder.Decode(&cases))
	}
	n := 0
	for _, c := range cases {
		truth := run(c, append(append([][]byte{}, c.Full...), c.Suffix...))
		actual := run(c, append(append([][]byte{}, c.Snapshot...), c.Suffix...))
		if !reflect.DeepEqual(truth, actual) || (c.Expected != nil && len(actual) != *c.Expected) || (c.Expected == nil && len(truth) == 0) {
			fmt.Fprintf(os.Stderr, "FAIL %s: full=%d keys, snapshot=%d keys\n", c.Name, len(truth), len(actual))
			shown := 0
			for k, v := range truth {
				if !reflect.DeepEqual(v, actual[k]) && shown < 5 {
					fmt.Fprintf(os.Stderr, "key %s: full=%v snapshot=%v\n", k, v, actual[k])
					shown++
				}
			}
			os.Exit(1)
		}
		n++
		fmt.Printf("PASS %s: %d indexed request keys\n", c.Name, len(actual))
	}
	fmt.Printf("PASS: %d real llm-d decoder/Pool/index comparisons\n", n)
}
