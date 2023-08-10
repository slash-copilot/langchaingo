package outputparser

import (
	"encoding/json"
	"fmt"

	"github.com/tmc/langchaingo/schema"
)

// ParseJSONError is the error type returned by output parsers.
type ParseJSONError struct {
	Text   string
	Reason string
}

func (e ParseJSONError) Error() string {
	return fmt.Sprintf("parse text %s. %s", e.Text, e.Reason)
}

const (
	// _structuredJSONFormatInstructionTemplate is a template for the format
	// instructions of the structuredJSON output parser.
	_structuredJSONFormatInstructionTemplate = "your input should strict follow json schema: \n\n{\n%s}\n" //nolint

	// _structuredJSONLineTemplate is a single line of the json schema in the
	// format instruction of the structuredJSON output parser. The fist verb is
	// the name, the second verb is the type and the third is a description of
	// what the field should contain.
	_structuredJSONLineTemplate = "\"%s\": %s // %s\n"
)

// ResponseJSONSchema is struct used in the structuredJSON output parser to describe
// how the llm should format its response. Name is a key in the parsed
// output map. Description is a description of what the value should contain.
type ResponseJSONSchema struct {
	Name        string
	Description string
}

// StructuredJSON is an output parser that parses the output of an llm into key value
// pairs. The name and description of what values the output of the llm should
// contain is stored in a list of response schema.
type StructuredJSON struct {
	ResponseJSONSchemas []ResponseJSONSchema
}

// NewStructuredJSON is a function that creates a new structuredJSON output parser from
// a list of response schemas.
func NewStructuredJSON(schema []ResponseJSONSchema) StructuredJSON {
	return StructuredJSON{
		ResponseJSONSchemas: schema,
	}
}

// Statically assert that StructuredJSON implement the OutputParser interface.
var _ schema.OutputParser[any] = StructuredJSON{}

// Parse parses the output of an llm into a map. If the output of the llm doesn't
// contain every filed specified in the response schemas, the function will return
// an error.
func (p StructuredJSON) parse(text string) (map[string]string, error) {
	// Remove the ```json that should be at the start of the text, and the ```
	// that should be at the end of the text.

	var parsed map[string]string
	err := json.Unmarshal([]byte(text), &parsed)
	if err != nil {
		return nil, err
	}

	// Validate that the parsed map contains all fields specified in the response
	// schemas.
	missingKeys := make([]string, 0)
	for _, rs := range p.ResponseJSONSchemas {
		if _, ok := parsed[rs.Name]; !ok {
			missingKeys = append(missingKeys, rs.Name)
		}
	}

	if len(missingKeys) > 0 {
		return nil, ParseJSONError{
			Text:   text,
			Reason: fmt.Sprintf("output is missing the following fields %v", missingKeys),
		}
	}

	return parsed, nil
}

func (p StructuredJSON) Parse(text string) (any, error) {
	return p.parse(text)
}

// ParseWithPrompt does the same as Parse.
func (p StructuredJSON) ParseWithPrompt(text string, _ schema.PromptValue) (any, error) {
	return p.parse(text)
}

// GetFormatInstructions returns a string explaining how the llm should format
// its response.
func (p StructuredJSON) GetFormatInstructions() string {
	jsonLines := ""
	for _, rs := range p.ResponseJSONSchemas {
		jsonLines += "\t" + fmt.Sprintf(
			_structuredJSONLineTemplate,
			rs.Name,
			"string", /* type of the filed*/
			rs.Description,
		)
	}

	return fmt.Sprintf(_structuredJSONFormatInstructionTemplate, jsonLines)
}

func (p StructuredJSON) GetFormatInstructionsWithPrompts(template string) string {
	jsonLines := ""
	for _, rs := range p.ResponseJSONSchemas {
		jsonLines += "\t" + fmt.Sprintf(
			_structuredJSONLineTemplate,
			rs.Name,
			"string", /* type of the filed*/
			rs.Description,
		)
	}
	return fmt.Sprintf(template, jsonLines)
}

// Type returns the type of the output parser.
func (p StructuredJSON) Type() string {
	return "structuredJSON_parser"
}
