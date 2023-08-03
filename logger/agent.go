package logger

import (
	"github.com/fatih/color"
	"github.com/tmc/langchaingo/schema"
)

func GetAgentLogger() AgentLogger {
	return &defaultAgentLogger{}
}

type defaultAgentLogger struct{}

var _ AgentLogger = defaultAgentLogger{}

// AgentThought implements AgentLogger.
func (defaultAgentLogger) AgentThought(action schema.AgentAction) {
	// Display banner
	banner("Agent Action")

	// Display thought
	message("Thought", action.Log, color.HiMagenta)
}
