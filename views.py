from django.shortcuts import render
import http.client
import json
from django.views.decorators.csrf import csrf_exempt


def enhance_prompt(prompt):
    # Here you can write your logic to improve or enhance the prompt
    enhanced_prompt = f"Design a landscape that embodies the spirit of the Kingdom of Artists. {prompt}. Cinematic, high detail."
    return enhanced_prompt

import requests

def generate_images(prompt):
    conn = http.client.HTTPSConnection("ai-text-to-image-generator-api.p.rapidapi.com")

    # Create the payload with the prompt
    payload = json.dumps({"inputs": prompt})

    headers = {
        'x-rapidapi-key': "ee7e10ded6mshdeae351dc6fd06ep18b9f7jsn3fa5d1589ee2",
        'x-rapidapi-host': "ai-text-to-image-generator-api.p.rapidapi.com",
        'Content-Type': "application/json"
    }

    try:
        # Send the POST request to the API
        conn.request("POST", "/realistic", payload, headers)
        
        # Get the response from the server
        res = conn.getresponse()
        data = res.read()
        
        # Decode the response from bytes to string and load it as JSON
        response_data = json.loads(data.decode("utf-8"))

        print("API Response:", response_data)  # Debugging print to see the full response

        # Extract the image URL from the response
        image_url = response_data.get('url', None)

        if image_url:
            return [image_url]
        else:
            print("No image URL found in the API response")
            return []

    except json.JSONDecodeError as json_error:
        print(f"Error decoding JSON response: {str(json_error)}")
        return []
    except Exception as e:
        print(f"General exception during API call: {str(e)}")
        return []

    
    
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def home(request):
    if request.method == 'POST':
        # Get the user prompt from POST data or request body
        try:
            user_prompt = request.POST.get('prompt', None) or request.body.decode('utf-8')
        except Exception as e:
            return JsonResponse({'error': f'Error processing request data: {str(e)}'}, status=400)
        
        if not user_prompt:
            return JsonResponse({'error': 'Invalid prompt'}, status=400)
        
        # Enhance the user prompt
        enhanced_prompt = enhance_prompt(user_prompt)

        try:
            # Generate images using the enhanced prompt
            images = generate_images(enhanced_prompt)
            print("Generated images:", images)  # Debug print

            if images:
                # Return JSON response for API calls
                response_data = {
                    'images': images,
                    'image_description': user_prompt,
                    'enhanced_prompt': enhanced_prompt
                }
                return JsonResponse(response_data)
            else:
                return JsonResponse({'error': 'Failed to generate images'}, status=500)
        except Exception as e:
            # Return detailed error for debugging
            return JsonResponse({'error': f'Internal Server Error: {str(e)}'}, status=500)
    
    return render(request, 'index.html')
