package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"

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

type Chat struct {
	client *openai.Client
	model  string
	Logger logger.LLMLogger
}

var (
	_ llms.ChatLLM       = (*Chat)(nil)
	_ llms.LanguageModel = (*Chat)(nil)
)

// Chat requests a chat response for the given messages.
func (o *Chat) Call(ctx context.Context, messages []schema.ChatMessage, options ...llms.CallOption) (*schema.AIChatMessage, error) { // nolint: lll
	r, err := o.Generate(ctx, [][]schema.ChatMessage{messages}, options...)
	if err != nil {
		return nil, err
	}
	if len(r) == 0 {
		return nil, ErrEmptyResponse
	}
	return r[0].Message, nil
}

func (o *Chat) Generate(ctx context.Context, messageSets [][]schema.ChatMessage, options ...llms.CallOption) ([]*llms.Generation, error) { // nolint:lll,cyclop
	opts := llms.CallOptions{MaxTokens: defaultMaxTokens}
	for _, opt := range options {
		opt(&opts)
	}

	model := opts.Model
	if len(model) == 0 {
		model = o.model
	}

	request := openai.ChatCompletionRequest{
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

	for _, fn := range opts.Functions {
		request.Functions = append(request.Functions, openai.FunctionDefinition{
			Name:        fn.Name,
			Description: fn.Description,
			Parameters:  fn.Parameters,
		})
		request.FunctionCall = llms.FunctionCallBehaviorAuto
	}

	generations := make([]*llms.Generation, 0, len(messageSets))

	openaiMessageSets := make([][]openai.ChatCompletionMessage, len(messageSets))
	for i, messageSet := range messageSets {
		msgs := make([]openai.ChatCompletionMessage, len(messageSet))
		for j, m := range messageSet {
			msg := openai.ChatCompletionMessage{
				Content: m.GetContent(),
			}
			typ := m.GetType()
			switch typ {
			case schema.ChatMessageTypeSystem:
				msg.Role = openai.ChatMessageRoleSystem
			case schema.ChatMessageTypeAI:
				msg.Role = openai.ChatMessageRoleAssistant
			case schema.ChatMessageTypeHuman:
				msg.Role = openai.ChatMessageRoleUser
			case schema.ChatMessageTypeGeneric:
				msg.Role = openai.ChatMessageRoleUser
			case schema.ChatMessageTypeFunction:
				msg.Role = openai.ChatMessageRoleFunction
			}
			msgs[j] = msg
		}
		openaiMessageSets[i] = msgs
	}

	for _, msgs := range openaiMessageSets {
		request.Messages = msgs

		v, _ := json.Marshal(request.Messages)
		o.Logger.LLMRequest(string(v))

		if request.Stream {
			generation, err := o.createChatCompletionStream(ctx, request, opts)
			if err != nil {
				o.Logger.LLMError(err)
				return nil, err
			}
			o.Logger.LLMResponse(generation.Text)
			generations = append(generations, generation)
		} else {
			generation, err := o.createChatCompletion(ctx, request)
			if err != nil {
				o.Logger.LLMError(err)
				return nil, err
			}
			o.Logger.LLMResponse(generation.Text)
			generations = append(generations, generation)
		}
	}
	return generations, nil
}

func (o *Chat) createChatCompletionStream(ctx context.Context, request openai.ChatCompletionRequest, opts llms.CallOptions) (*llms.Generation, error) { // nolint:lll
	stream, err := o.client.CreateChatCompletionStream(ctx, request)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var (
		text         = ""
		finishReason = ""
		functionCall *schema.FunctionCall
	)

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

		content := resp.Choices[0].Delta.Content
		err = opts.StreamingFunc(ctx, []byte(content))
		if err != nil {
			return nil, err
		}

		text += content
		finishReason = string(resp.Choices[0].FinishReason)

		if resp.Choices[0].FinishReason == openai.FinishReasonFunctionCall {
			functionCall = &schema.FunctionCall{
				Name:      resp.Choices[0].Delta.FunctionCall.Name,
				Arguments: resp.Choices[0].Delta.FunctionCall.Arguments,
			}
		}
	}

	return &llms.Generation{
		Message: &schema.AIChatMessage{
			Content:      text,
			FunctionCall: functionCall,
		},
		GenerationInfo: map[string]any{
			"FinishReason": finishReason,
		},
	}, nil
}

func (o *Chat) createChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (*llms.Generation, error) { // nolint:lll
	resp, err := o.client.CreateChatCompletion(ctx, request)
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, ErrEmptyResponse
	}

	text := resp.Choices[0].Message.Content
	finishReason := string(resp.Choices[0].FinishReason)
	var functionCall *schema.FunctionCall

	if resp.Choices[0].FinishReason == openai.FinishReasonFunctionCall {
		functionCall = &schema.FunctionCall{
			Name:      resp.Choices[0].Message.FunctionCall.Name,
			Arguments: resp.Choices[0].Message.FunctionCall.Arguments,
		}
	}

	return &llms.Generation{
		Message: &schema.AIChatMessage{
			Content:      text,
			FunctionCall: functionCall,
		},
		GenerationInfo: map[string]any{
			"CompletionTokens": resp.Usage.CompletionTokens,
			"PromptTokens":     resp.Usage.PromptTokens,
			"TotalTokens":      resp.Usage.TotalTokens,
			"FinishReason":     finishReason,
		},
	}, nil
}

func (o *Chat) GetNumTokens(text string) int {
	return llms.CountTokens(o.model, text)
}

func (o *Chat) GeneratePrompt(ctx context.Context, promptValues []schema.PromptValue, options ...llms.CallOption) (llms.LLMResult, error) { //nolint:lll
	return llms.GenerateChatPrompt(ctx, o, promptValues, options...)
}

// New returns a new OpenAI LLM.
func New(opts ...Option) (*LLM, error) {
	options := &options{
		token:   os.Getenv(tokenEnvVarName),
		model:   os.Getenv(modelEnvVarName),
		baseURL: os.Getenv(baseURLEnvVarName),
		logger:  logger.GetLLMLogger(),
	}

	for _, opt := range opts {
		opt(options)
	}

	if len(options.model) == 0 {
		options.model = defaultCompletionModel
	}

	c, err := newClient(options)
	return &LLM{
		model:  options.model,
		client: c,
		Logger: options.logger,
	}, err
}

// NewChat returns a new OpenAI chat LLM.
func NewChat(opts ...Option) (*Chat, error) {
	options := &options{
		token:   os.Getenv(tokenEnvVarName),
		model:   os.Getenv(modelEnvVarName),
		baseURL: os.Getenv(baseURLEnvVarName),
		logger:  logger.GetLLMLogger(),
	}

	for _, opt := range opts {
		opt(options)
	}

	if len(options.model) == 0 {
		options.model = defaultChatModel
	}

	c, err := newClient(options)
	return &Chat{
		model:  options.model,
		client: c,
		Logger: options.logger,
	}, err
}

func newClient(options *options) (*openai.Client, error) {
	if len(options.token) == 0 {
		return nil, ErrMissingToken
	}
	config := openai.DefaultConfig(options.token)
	if options.baseURL != "" {
		config.BaseURL = options.baseURL
	}
	client := openai.NewClientWithConfig(config)
	return client, nil
}

// NewAzure returns a new Azure OpenAI LLM .
func NewAzure(opts ...Option) (*LLM, error) {
	c, err := newAzureClient(opts...)
	return &LLM{
		client: c,
	}, err
}

// NewAzureChat returns a new OpenAI chat LLM.
func NewAzureChat(opts ...Option) (*Chat, error) {
	c, err := newAzureClient(opts...)
	return &Chat{
		client: c,
	}, err
}

func newAzureClient(opts ...Option) (*openai.Client, error) {
	options := &options{
		token:        os.Getenv(tokenEnvVarName),
		model:        os.Getenv(modelEnvVarName),
		baseURL:      os.Getenv(baseURLEnvVarName),
		organization: os.Getenv(organizationEnvVarName),
		apiType:      APITypeAzure,
		apiVersion:   DefaultAPIVersion,
	}

	for _, opt := range opts {
		opt(options)
	}

	if len(options.token) == 0 {
		return nil, ErrMissingToken
	}

	config := openai.DefaultAzureConfig(options.token, options.baseURL)
	config.APIVersion = options.apiVersion
	config.OrgID = options.organization

	return openai.NewClientWithConfig(config), nil
}
