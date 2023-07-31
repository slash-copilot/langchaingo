package openai

import (
	"os"

	"github.com/sashabaranov/go-openai"
)

// newClient is wrapper for openaiclient internal package.
func newClient(opts ...Option) (*openai.Client, error) {
	options := &options{
		token:        os.Getenv(tokenEnvVarName),
		model:        os.Getenv(modelEnvVarName),
		organization: os.Getenv(organizationEnvVarName),
		apiType:      APIType(openai.APITypeOpenAI),
	}

	for _, opt := range opts {
		opt(options)
	}

	if len(options.token) == 0 {
		return nil, ErrMissingToken
	}

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
