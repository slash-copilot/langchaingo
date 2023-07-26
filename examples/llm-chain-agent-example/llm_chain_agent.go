package main

import (
	"context"
	"fmt"
	"os"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/tools"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	// We can construct an LLMChain from a PromptTemplate and an LLM.
	llm, err := openai.New(
		openai.WithBaseURL(os.Getenv("OPENAI_BASE_URL")),
		openai.WithToken(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		return err
	}

	executor, err := agents.Initialize(
		llm,
		[]tools.Tool{tools.Calculator{}},
		agents.ConversationalReactDescription,
		agents.WithMemory(memory.NewBuffer()),
	)

	if err != nil {
		return err
	}

	res1, _ := chains.Run(context.Background(), executor, "Hi! my name is Bob and the year I was born is 1987")

	fmt.Println(res1)

	res2, _ := chains.Run(context.Background(), executor, "What is the year I was born times 34")

	fmt.Println(res2)

	return nil
}
