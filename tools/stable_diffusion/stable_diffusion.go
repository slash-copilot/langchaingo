package stable_diffusion

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/tmc/langchaingo/outputparser"
	"github.com/tmc/langchaingo/tools"
	"github.com/tmc/langchaingo/tools/stable_diffusion/internal"
)

var ErrMissingURL = errors.New("missing `SD_WEBUI_URL` environment variable")

type Tool struct {
	SDWebUIClient    *internal.SDWebUIClient
	structuredPrompt outputparser.StructuredJSON
	options          *createOptions
}

var _ tools.Tool = Tool{}

type createOptions struct {
	URL        string
	Iterations int
	Width      int
	Height     int
	OutputPath string
	StaticPath string
}

func DefaultCreateOptions() *createOptions {
	return &createOptions{
		URL:        os.Getenv("SD_WEBUI_URL"),
		Iterations: 20,
		Width:      512,
		Height:     512,
		OutputPath: "./images",
		StaticPath: "/static/images",
	}
}

type CreateSDOption func(*createOptions)

func WithURL(url string) func(*createOptions) {
	return func(o *createOptions) {
		o.URL = url
	}
}

func WithIterations(iterations int) func(*createOptions) {
	return func(o *createOptions) {
		o.Iterations = iterations
	}
}

func WithWidth(width int) func(*createOptions) {
	return func(o *createOptions) {
		o.Width = width
	}
}

func WithHeight(height int) func(*createOptions) {
	return func(o *createOptions) {
		o.Height = height
	}
}

func WithOutputPath(outputPath string) func(*createOptions) {
	return func(o *createOptions) {
		o.OutputPath = outputPath
	}
}

func WithStaticPath(staticPath string) func(*createOptions) {
	return func(o *createOptions) {
		o.StaticPath = staticPath
	}
}

// New creates a new stable_diffusion tool to generate images.
func New(opts ...CreateSDOption) (*Tool, error) {
	options := DefaultCreateOptions()

	for _, opt := range opts {
		opt(options)
	}

	if options.URL == "" {
		return nil, ErrMissingURL
	}

	client := internal.NewSDWebUIClient()
	client.SetAPIUrl(options.URL)

	return &Tool{
		SDWebUIClient: client,
		structuredPrompt: outputparser.NewStructuredJSON([]outputparser.ResponseJSONSchema{
			{
				Name:        "prompt",
				Description: "Required, Detailed keywords to describe the subject, using at least 7 keywords to accurately describe the image, separated by comma",
			},
			{
				Name:        "negativePrompt",
				Description: "Required, Detailed Keywords we want to exclude from the final image, using at least 7 keywords to accurately describe the image, separated by comma",
			},
		}),
		options: options,
	}, nil
}
func (t Tool) Name() string {
	return "stable-diffusion"
}

func (t Tool) Description() string {
	return fmt.Sprintf(`
	You can generate images with 'stable-diffusion'. This tool is exclusively for visual content.
Guidelines:
1. Visually describe the moods, details, structures, styles, and/or proportions of the image. Remember, the focus is on visual attributes.
2. Craft your input by "showing" and not "telling" the imagery. Think in terms of what you'd want to see in a photograph or a painting.
3. %s,  
4. Here is an example call for generating a realistic portrait photo of a man:
	 {
		"prompt": "photo of a man in black clothes, half body, high detailed skin, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
		"negativePrompt": "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, out of frame, low quality, ugly, mutation, deformed"
	 }
	`, t.structuredPrompt.GetFormatInstructions())
}

func (t Tool) Call(ctx context.Context, input string) (string, error) {
	values, err := t.structuredPrompt.Parse(input)

	if err != nil {
		return "", fmt.Errorf("stable-diffusion: invalid input format, %v", err)
	}

	valuesMap, ok := values.(map[string]string)

	if !ok {
		return "", fmt.Errorf("stable-diffusion: invalid input format, %v", err)
	}

	payload := internal.TXT2IMGReq{
		Prompt:         valuesMap["prompt"],
		NegativePrompt: valuesMap["negativePrompt"],
		Steps:          20,
		Width:          512,
		Height:         512,
		SamplerName:    "DPM++ SDE Karras",
	}

	base64ImgStr, err := t.SDWebUIClient.Text2ImgWithCustomPrompt(&payload)

	if err != nil {
		return "", err
	}

	data, err := base64.StdEncoding.DecodeString(base64ImgStr)

	if err != nil {
		return "", err
	}

	imageName := fmt.Sprintf("%d.png", time.Now().UnixNano())
	outputPath := filepath.Join(t.options.OutputPath, imageName)
	staticPath := filepath.Join(t.options.StaticPath, imageName)

	if _, err := os.Stat(filepath.Dir(outputPath)); os.IsNotExist(err) {
		err = os.MkdirAll(filepath.Dir(outputPath), os.ModePerm)
		if err != nil {
			return "", err
		}
	}

	err = os.WriteFile(outputPath, data, 0644)

	if err != nil {
		return "", err
	}

	return fmt.Sprintf("![generated image](%s)", staticPath), nil
}
