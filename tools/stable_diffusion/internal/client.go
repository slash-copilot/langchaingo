package internal

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
)

type SDWebUIClient struct {
	Client        *http.Client
	ApiUrl        string
	BasicAuthUser string
	BasicAuthPass string
}

func NewSDWebUIClient() *SDWebUIClient {
	return &SDWebUIClient{
		Client: &http.Client{},
	}
}

/*
set the api endpoint url
eg. http://127.0.0.1:7860
eg. http://user:pass@localhost:7860
*/
func (c *SDWebUIClient) SetAPIUrl(apiUrl string) error {
	url, err := url.ParseRequestURI(apiUrl)

	if err != nil { // the api url is wrong
		return err
	}

	c.ApiUrl = url.Scheme + "://" + url.Host

	if url.User != nil {
		c.BasicAuthUser = url.User.Username()
		c.BasicAuthPass, _ = url.User.Password()
	}

	return nil
}

/*
Set Proxy
eg. http://127.0.0.1:8080
*/
func (c *SDWebUIClient) SetProxy(proxyUrl string) error {
	proxy, err := url.ParseRequestURI(proxyUrl)
	if err != nil {
		return err
	}
	c.Client.Transport = &http.Transport{Proxy: http.ProxyURL(proxy)}
	return nil
}

func (c *SDWebUIClient) Text2ImgWithDefaultPrompt(prompt string) (string, error) {
	defaultPayload := getDefaultDataTXT2IMGReq()
	defaultPayload.Prompt = defaultPayload.Prompt + prompt
	b, err := json.Marshal(defaultPayload)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequest("POST", c.ApiUrl+"/sdapi/v1/txt2img", bytes.NewBuffer(b))

	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")

	if c.BasicAuthUser != "" {
		req.SetBasicAuth(c.BasicAuthUser, c.BasicAuthPass)
	}

	resp, err := c.Client.Do(req)

	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	res := TXT2IMGResp{}
	err = json.Unmarshal(body, &res)
	return res.Images[0], err
}

func (c *SDWebUIClient) Text2ImgWithCustomPrompt(txt2imgReq *TXT2IMGReq) (string, error) {
	b, err := json.Marshal(txt2imgReq)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", c.ApiUrl+"/sdapi/v1/txt2img", bytes.NewBuffer(b))

	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")

	if c.BasicAuthUser != "" {
		req.SetBasicAuth(c.BasicAuthUser, c.BasicAuthPass)
	}

	resp, err := c.Client.Do(req)

	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	res := TXT2IMGResp{}
	err = json.Unmarshal(body, &res)
	return res.Images[0], err
}

/*
Set the initial image which should be base64 encoded
*/
func (c *SDWebUIClient) Img2Img(img string) (string, error) {
	i := IMG2IMGReq{
		InitImages: []string{""},
	}
	i.InitImages[0] = img
	b, err := json.Marshal(i)
	if err != nil {
		return "", err
	}
	resp, err := c.Client.Post(c.ApiUrl+"/sdapi/v1/img2img", "application/json", bytes.NewBuffer(b))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	res := IMG2IMGResp{}
	err = json.Unmarshal(body, &res)
	return res.Images[0], err
}

/*
Extras Single Image
*/
func (c *SDWebUIClient) ExtrasSingleImage(req ExtrasSingleImageReq) (ExtrasSingleImageResp, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return ExtrasSingleImageResp{}, err
	}
	resp, err := c.Client.Post(c.ApiUrl+"/sdapi/v1/extra-single-image", "application/json", bytes.NewBuffer(b))
	if err != nil {
		return ExtrasSingleImageResp{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	res := ExtrasSingleImageResp{}
	err = json.Unmarshal(body, &res)
	return res, err
}

/*
Get Config
*/
func (c *SDWebUIClient) GetConfig() (ConfigResp, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/options")
	if err != nil {
		return ConfigResp{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	config := ConfigResp{}
	err = json.Unmarshal(body, &config)
	return config, err
}

/*
Set Stable Diffusion Checkpoint
@parameter cp: the name of checkpoint in Stable Diffusion Checkpoint int WebUI
*/
func (c *SDWebUIClient) SetStableDiffusionCheckpoint(cp string) error {
	req := CheckpointReq{
		SdModelCheckpoint: cp,
	}
	b, err := json.Marshal(req)
	if err != nil {
		return err
	}
	_, err = c.Client.Post(c.ApiUrl+"/sdapi/v1/options", "application/json", bytes.NewBuffer(b))
	return err

}

/*
Get Stable Diffusion Checkpoint
*/
func (c *SDWebUIClient) GetStableDiffusionCheckpoint() (string, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/options")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	config := ConfigResp{}
	err = json.Unmarshal(body, &config)
	return config.SDModelCheckpoint, err
}

/*
Get Memory Status
*/
func (c *SDWebUIClient) GetMemory() (MemStatus, error) {

	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/memory")
	if err != nil {
		return MemStatus{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	status := MemStatus{}
	err = json.Unmarshal(body, &status)
	return status, err
}

/*
Get Sd Models
*/
func (c *SDWebUIClient) GetSdModels() (SDModels, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/sd-models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	models := SDModels{}
	err = json.Unmarshal(body, &models)
	return models, err
}

/*
Get Prompt Styles
*/
func (c *SDWebUIClient) GetPromptStyles() (PromptStyles, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/prompt-styles")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	styles := PromptStyles{}
	err = json.Unmarshal(body, &styles)
	return styles, err
}

/*
Get Realesrgan Models
*/
func (c *SDWebUIClient) GetRealesrganModels() (RealesrganModels, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/realesrgan-models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	models := RealesrganModels{}
	err = json.Unmarshal(body, &models)
	return models, err
}

/*
Get Face Restorers
*/
func (c *SDWebUIClient) GetFaceRestorers() (FaceRestorers, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/face-restorers")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	restorers := FaceRestorers{}
	err = json.Unmarshal(body, &restorers)
	return restorers, err
}

/*
Get Embeddings
!Not stable
*/
func (c *SDWebUIClient) GetEmbeddings() (Embeddings, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/embeddings")
	if err != nil {
		return Embeddings{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	embeddings := Embeddings{}
	err = json.Unmarshal(body, &embeddings)
	return embeddings, err
}

/*
Get Hypernetworks
*/
func (c *SDWebUIClient) GetHypernetworks() (Hypernetworks, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/hypernetworks")
	if err != nil {
		return Hypernetworks{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	hypernetworks := Hypernetworks{}
	err = json.Unmarshal(body, &hypernetworks)
	return hypernetworks, err
}

/*
Get Upscalers
*/
func (c *SDWebUIClient) GetUpscalers() (Upscalers, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/upscalers")
	if err != nil {
		return Upscalers{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	upscalers := Upscalers{}
	err = json.Unmarshal(body, &upscalers)
	return upscalers, err
}

/*
Get Cmd Flags
*/
func (c *SDWebUIClient) GetCmdFlags() (CmdFlags, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/sdapi/v1/cmd-flags")
	if err != nil {
		return CmdFlags{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	cmdFlags := CmdFlags{}
	err = json.Unmarshal(body, &cmdFlags)
	return cmdFlags, err
}

// Get Current User
func (c *SDWebUIClient) GetCurrentUser() (string, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/user")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	return string(body), err
}

// Login Check
func (c *SDWebUIClient) LoginCheck() (string, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/login_check")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	return string(body), err
}

// Get Token
func (c *SDWebUIClient) GetToken() (Token, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/token")
	if err != nil {
		return Token{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	token := Token{}
	err = json.Unmarshal(body, &token)
	return token, err
}

// App Id
func (c *SDWebUIClient) AppId() (AppId, error) {
	resp, err := c.Client.Get(c.ApiUrl + "/app_id")
	if err != nil {
		return AppId{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	appId := AppId{}
	err = json.Unmarshal(body, &appId)
	return appId, err
}

// Reset Iterator
func (c *SDWebUIClient) Reset(req ResetReq) (ResetResp, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return ResetResp{}, err
	}
	resp, err := c.Client.Post(c.ApiUrl+"/reset", "application/json", bytes.NewBuffer(b))
	if err != nil {
		return ResetResp{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	res := ResetResp{}
	err = json.Unmarshal(body, &res)
	return res, err
}

//!!! Not implemented yet.

func (c *SDWebUIClient) FetchFile() {
	panic("not implemented")
}

func (c *SDWebUIClient) TrainHypernetwork() {
	panic("not implemented")
}

func (c *SDWebUIClient) TrainEmbedding() {
	panic("not implemented")
}

func (c *SDWebUIClient) Preprocess() {
	panic("not implemented")
}

func (c *SDWebUIClient) CreateHypernetwork() {
	panic("not implemented")
}

func (c *SDWebUIClient) CreateEmbedding() {
	panic("not implemented")
}

func (c *SDWebUIClient) RefreshCheckpoints() {
	panic("not implemented")
}

func (c *SDWebUIClient) GetSamples() {
	panic("not implemented")
}

func (c *SDWebUIClient) SetConfig() {
	panic("not implemented")
}

func (c *SDWebUIClient) GetConifg() {
	panic("not implemented")
}

func (c *SDWebUIClient) Skip() {
	panic("not implemented")
}

func (c *SDWebUIClient) Interrupt() {
	panic("not implemented")
}

func (c *SDWebUIClient) Interrogate() {
	panic("not implemented")
}

func (c *SDWebUIClient) Progress() {
	panic("not implemented")
}

func (c *SDWebUIClient) PngInfo() {
	panic("not implemented")
}

func (c *SDWebUIClient) ExtrasBatchImages() {
	panic("not implemented")
}

func (c *SDWebUIClient) RobotsTxt() {
	panic("not implemented")
}

func (c *SDWebUIClient) StartupEvents() {
	panic("not implemented")
}

func (c *SDWebUIClient) GetQueueStatus() {
	panic("not implemented")
}
