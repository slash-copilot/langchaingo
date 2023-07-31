package openai

import (
	"context"
	"errors"
	"io"

	"github.com/sashabaranov/go-openai"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/logger"
	"github.com/tmc/langchaingo/schema"
)

const (
	defaultChatModel       = "gpt-3.5-turbo"
	defaultCompletionModel = "text-davinci-003"
	defaultEmbeddingModel  = "text-embedding-ada-002"

	defaultMaxTokens = 1024
)

var (
	ErrEmptyResponse = errors.New("no response")
	ErrMissingToken  = errors.New("missing the OpenAI API key, set it in the OPENAI_API_KEY environment variable")

	ErrUnexpectedResponseLength = errors.New("unexpected length of response")
	ErrUnexpectedEmbeddingModel = errors.New("unexpected embedding model")
)

type LLM struct {
	model  string
	client *openai.Client
	Logger logger.LLMLogger
}

var (
	_ llms.LLM           = (*LLM)(nil)
	_ llms.LanguageModel = (*LLM)(nil)
)

// New returns a new OpenAI LLM.
func New(opts ...Option) (*LLM, error) {
	c, err := newClient(opts...)
	return &LLM{
		client: c,
		model:  defaultCompletionModel,
		Logger: logger.GetLLMLogger(),
	}, err
}

// Call requests a completion for the given prompt.
func (o *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	r, err := o.Generate(ctx, []string{prompt}, options...)
	if err != nil {
		return "", err
	}
	if len(r) == 0 {
		return "", ErrEmptyResponse
	}
	return r[0].Text, nil
}

func (o *LLM) Generate(ctx context.Context, prompts []string, options ...llms.CallOption) ([]*llms.Generation, error) {
	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	model := opts.Model
	if len(model) == 0 {
		model = o.model
	}

	generations := make([]*llms.Generation, 0, len(prompts))

	request := openai.CompletionRequest{
		Model:            model,
		MaxTokens:        opts.MaxTokens,
		Temperature:      float32(opts.Temperature),
		TopP:             float32(opts.TopP),
		Stream:           opts.StreamingFunc != nil,
		Stop:             opts.StopWords,
		N:                opts.N,
		FrequencyPenalty: float32(opts.FrequencyPenalty),
		PresencePenalty:  float32(opts.PresencePenalty),
	}

	for _, prompt := range prompts {
		o.Logger.LLMRequest(prompt)
		request.Prompt = prompt

		if request.Stream {
			generation, err := o.createCompletionStream(ctx, request, opts)
			if err != nil {
				o.Logger.LLMResponse(err.Error())
				return nil, err
			}
			o.Logger.LLMResponse(generation.Text)
			generations = append(generations, generation)
		} else {
			generation, err := o.createCompletion(ctx, request)
			if err != nil {
				o.Logger.LLMResponse(err.Error())
				return nil, err
			}
			o.Logger.LLMResponse(generation.Text)
			generations = append(generations, generation)
		}
	}

	return generations, nil
}

func (o *LLM) createCompletionStream(ctx context.Context, request openai.CompletionRequest, opts llms.CallOptions) (*llms.Generation, error) { // nolint:lll
	stream, err := o.client.CreateCompletionStream(ctx, request)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	output := ""
	finishReason := ""
	for {
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}

		if len(resp.Choices) == 0 {
			return nil, ErrEmptyResponse
		}

		text := resp.Choices[0].Text
		err = opts.StreamingFunc(ctx, []byte(text))
		if err != nil {
			return nil, err
		}

		output += text
		finishReason = resp.Choices[0].FinishReason
	}

	return &llms.Generation{
		Text: output,
		GenerationInfo: map[string]any{
			"FinishReason": finishReason,
		},
	}, nil
}

func (o *LLM) createCompletion(ctx context.Context, request openai.CompletionRequest) (*llms.Generation, error) {
	resp, err := o.client.CreateCompletion(ctx, request)
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, ErrEmptyResponse
	}

	text := resp.Choices[0].Text
	finishReason := resp.Choices[0].FinishReason
	return &llms.Generation{
		Text: text,
		GenerationInfo: map[string]any{
			"CompletionTokens": resp.Usage.CompletionTokens,
			"PromptTokens":     resp.Usage.PromptTokens,
			"TotalTokens":      resp.Usage.TotalTokens,
			"FinishReason":     finishReason,
		},
	}, nil
}

func (o *LLM) GeneratePrompt(ctx context.Context, promptValues []schema.PromptValue, options ...llms.CallOption) (llms.LLMResult, error) { //nolint:lll
	return llms.GeneratePrompt(ctx, o, promptValues, options...)
}

func (o *LLM) GetNumTokens(text string) int {
	return llms.CountTokens(o.model, text)
}

// CreateEmbedding creates embeddings for the given input texts.
func (o *LLM) CreateEmbedding(ctx context.Context, model string, inputTexts []string) ([][]float64, error) {
	if len(model) == 0 {
		model = defaultEmbeddingModel
	}

	embeddingModel, ok := stringToEmbeddingModel[model]
	if !ok {
		return nil, ErrUnexpectedEmbeddingModel
	}
	resp, err := o.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: embeddingModel,
		Input: inputTexts,
	})
	if err != nil {
		return [][]float64{}, err
	}

	data := resp.Data
	if len(data) == 0 {
		return [][]float64{}, ErrEmptyResponse
	}
	if len(inputTexts) != len(data) {
		return [][]float64{}, ErrUnexpectedResponseLength
	}

	embeddings := make([][]float64, len(data))
	for i := range data {
		embedding := make([]float64, len(data[i].Embedding))
		for j := range data[i].Embedding {
			embedding[j] = float64(data[i].Embedding[j])
		}
		embeddings[i] = embedding
	}
	return embeddings, nil
}
