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
	structuredPrompt outputparser.Structured
}

var _ tools.Tool = Tool{}

// New creates a new serpapi tool to search on internet.
func New() (*Tool, error) {
	url := os.Getenv("SD_WEBUI_URL")
	if url == "" {
		return nil, ErrMissingURL
	}
	client := internal.NewSDWebUIClient()
	client.SetAPIUrl(url)

	return &Tool{
		SDWebUIClient: client,
		structuredPrompt: outputparser.NewStructured([]outputparser.ResponseSchema{
			{
				Name:        "prompt",
				Description: "required, detailed keywords to describe the subject, separated by commas",
			},
			{
				Name:        "negativePrompt",
				Description: "required, detailed keywords we want to exclude from the final, separated by commas",
			},
		}),
	}, nil
}
func (t Tool) Name() string {
	return "stable-diffusion"
}

func (t Tool) Description() string {

	return fmt.Sprintf(`
	You can generate images with 'stable-diffusion'. This tool is exclusively for visual content.
	call it with the following format:
		%s,
	- here is an an example prompt for generating a realistic portrait photo of a man:
	 {
		"prompt": "photo of a man in black clothes, half body, high detailed skin, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3 ",
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
	outputPath := filepath.Join(".", "images", imageName)

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

	return fmt.Sprintf("![generated image](/%s)", outputPath), nil
}
