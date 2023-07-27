package logger

import "github.com/fatih/color"

func GetLLMLogger() LLMLogger {
	return &defaultLLMLogger{}
}

type defaultLLMLogger struct{}

var _ LLMLogger = defaultLLMLogger{}

// LLMError implements LLMLogger.
func (defaultLLMLogger) LLMError(err error) {
	// Display banner
	banner("LLM Query")

	// Display error
	message("Received error", err.Error(), color.Red)
}

// LLMRequest implements LLMLogger.
func (defaultLLMLogger) LLMRequest(msg string) {
	// Display banner
	banner("LLM Query")

	// Display question
	message("Submitted query", msg, color.Cyan)
}

// LLMResponse implements LLMLogger.
func (defaultLLMLogger) LLMResponse(msg string) {
	// Display banner
	banner("LLM Query")
	// Display answer
	message("Received response", msg, color.Green)
}
