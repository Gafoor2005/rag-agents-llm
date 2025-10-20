
<br>

## Trying Out The Foundation Model Endpoints

In this section, you will finally get to interact with your LLM endpoints! 

**From Your Own Environment**: You would want to go to [`build.nvidia.com`](https://build.nvidia.com/) and find a model you'd like to use. For example, you could go to [**the MistralAI's Mixtral-8x7b model**](https://build.nvidia.com/mistralai/mixtral-8x7b-instruct) to see an example of how to use the model, links for further readings, and some buttons like "Apply To Self-Host" and "Get API Key."

- Clicking **"Apply To Self-Host"** will guide you to information about NVIDIA Microservices and give you some avenues to sign up (i.e. early access/NVIDIA AI Enterprise pathway) or enter a notification list (General Access pathway).

- Clicking **"Get API Key"** will generate an API key starting with "nvapi-" which you can provide to the API endpoints via a network request!

If you were to do this, you would need to add the API key to the notebook like so:

```python
import os
os.environ["NVIDIA_API_KEY"] = "nvapi-..."
```

