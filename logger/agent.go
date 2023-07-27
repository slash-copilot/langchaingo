package logger

import "github.com/fatih/color"

func GetAgentLogger() AgentLogger {
	return &defaultAgentLogger{}
}

type defaultAgentLogger struct{}

var _ AgentLogger = defaultAgentLogger{}

// AgentThought implements AgentLogger.
func (defaultAgentLogger) AgentThought(msg string) {
	// Display banner
	banner("Agent Action")

	// Display thought
	message("Thought", msg, color.HiMagenta)
}
