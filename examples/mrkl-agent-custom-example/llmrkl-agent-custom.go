package main

import (
	"context"
	"fmt"
	"os"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
	"github.com/tmc/langchaingo/tools/stable_diffusion"
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

	stableDiffusionTool, err := stable_diffusion.New()

	if err != nil {
		return err
	}

	executor, err := agents.Initialize(
		llm,
		[]tools.Tool{stableDiffusionTool},
		agents.ConversationalReactDescription,
		agents.WithMaxIterations(3),
	)

	if err != nil {
		return err
	}

	res1, err := chains.Run(context.Background(), executor, "Hi! I want to draw a picture of a dog.")

	if err != nil {
		return err
	}

	fmt.Println(res1)

	return nil
}
