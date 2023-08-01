package stable_diffusion

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmc/langchaingo/tools"
	"github.com/tmc/langchaingo/tools/stable_diffusion/internal"
)

var ErrMissingURL = errors.New("missing `SD_WEBUI_URL` environment variable")

type Tool struct {
	SDWebUIClient *internal.SDWebUIClient
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
	}, nil
}

func (t Tool) Name() string {
	return "stable-diffusion"
}

func (t Tool) Description() string {
	return `
	You can generate images with 'stable-diffusion'. This tool is exclusively for visual content.
	Guidelines:
	- Visually describe the moods, details, structures, styles, and/or proportions of the image. Remember, the focus is on visual attributes.
	- Craft your input by "showing" and not "telling" the imagery. Think in terms of what you'd want to see in a photograph or a painting.
	- It's best to follow this format for image creation:
	"detailed keywords to describe the subject, separated by comma | keywords we want to exclude from the final image"
	- Here's an example prompt for generating a realistic portrait photo of a man:
	"photo of a man in black clothes, half body, high detailed skin, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3 | semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, out of frame, low quality, ugly, mutation, deformed"
	- Generate images only once per human query unless explicitly requested by the user"`
}

func (t Tool) Call(ctx context.Context, input string) (string, error) {
	input = strings.Trim(input, "\n")
	inputs := strings.Split(input, "|")

	if len(inputs) != 2 {
		return "", fmt.Errorf("stable-diffusion: invalid input format")
	}

	payload := internal.TXT2IMGReq{
		Prompt:         inputs[0],
		NegativePrompt: inputs[1],
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
