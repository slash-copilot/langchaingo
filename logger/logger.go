package logger

type LLMLogger interface {
	LLMRequest(msg string)
	LLMResponse(msg string)
	LLMError(err error)
}

type AgentLogger interface {
	AgentThought(msg string)
}
