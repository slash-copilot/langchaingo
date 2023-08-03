package logger

import "github.com/tmc/langchaingo/schema"

type LLMLogger interface {
	LLMRequest(msg string)
	LLMResponse(msg string)
	LLMError(err error)
}

type AgentLogger interface {
	AgentThought(schema.AgentAction)
}
