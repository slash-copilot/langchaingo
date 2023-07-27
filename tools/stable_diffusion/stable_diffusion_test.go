package stable_diffusion

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSDClient(t *testing.T) {
	t.Parallel()

	tool, err := New()
	assert.NoError(t, err)

	result, err := tool.Call(context.Background(), "photo of a man in black clothes, half body, high detailed skin, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3 | semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, out of frame, low quality, ugly, mutation, deformed")

	assert.NoError(t, err)

	assert.NotEmpty(t, result)
}
