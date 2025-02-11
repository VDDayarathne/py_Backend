from django.shortcuts import render, redirect
import json
import base64
import os
import re
import openai
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import google.generativeai as genai
from .models import GeneratedImage, UserProgress, PromptFeedback
import requests
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserRegisterForm
from django.conf import settings
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator
from django.contrib.auth import logout
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.decorators import login_required

db = settings.MONGO_DB

GEMINI_API_KEY = 'AIzaSyCBTHsGoHzbNnafqt_rJEJNTthwfYhaEaE'
CLIPDROP_API_KEY = '0ebcadff5f339e8780c8aecfdeb96a600f68ba12189cced43406c9a0bc93752f1554211f78df6682cf4c3a615e29bda5'  
IMAGEN_API_KEY = 'AIzaSyCBTHsGoHzbNnafqt_rJEJNTthwfYhaEaE'  
STABILITY_AI_API_KEY = 'sk-Ezo9a98HEdeSzopjxgQrAiLVFMWLiQYcn3iJrrPKkgR8T0m7'
META_AI_API_KEY = ''

# Configure Imagen API
genai.configure(api_key=IMAGEN_API_KEY)



# Home Page
def index(request):
    return render(request, 'base/index.html')


@csrf_exempt
def register(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)  
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    form = UserRegisterForm(data)
    if form.is_valid():
        user = form.save()

        refresh = RefreshToken.for_user(user)
        access_token = str(refresh.access_token)
        refresh_token = str(refresh)

        return JsonResponse({"message": f"Account created successfully for {user.username}!",
            "access": access_token,
            "refresh": refresh_token,
        }, status=201)

    return JsonResponse({"errors": form.errors}, status=400)

 
@csrf_exempt
def login_view(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
        username = data.get("username")
        password = data.get("password")
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    user = authenticate(request, username=username, password=password)
    if user:
        refresh = RefreshToken.for_user(user)
        access_token = str(refresh.access_token)
        refresh_token = str(refresh)
        login(request, user)
        return JsonResponse({"message": f"Welcome, {user.username}!",
            "access": access_token,
            "refresh": refresh_token,
        }, status=200)

    return JsonResponse({"error": "Invalid username or password"}, status=400)

@csrf_exempt  
def user_logout(request):
    if request.method == "POST":
        logout(request)
        return JsonResponse({"message": "User logged out successfully!"}, status=200)

    return JsonResponse({"error": "Invalid request method"}, status=405)





# enhance the prompt using Gemini API
def enhance_prompt_with_gemini(user_prompt, task_objective, task_description):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}'
    headers = {'Content-Type': 'application/json'}
    data = {
        'contents': [{
            'parts': [{
                'text': f"{task_objective}. {task_description}. {user_prompt}"
            }]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        gemini_response = response.json()
        enhanced_prompt = gemini_response['candidates'][0]['content']['parts'][0]['text']
        return enhanced_prompt
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise Exception(f"Failed to connect to Gemini API: {e}")
    except (KeyError, IndexError) as e:
        print(f"Unexpected response structure: {e}")
        raise Exception("Unexpected response structure from Gemini API.")





def enhance_prompt_with_openai(user_prompt, task_objective, task_description):
    prompt = f"{task_objective}. {task_description}. {user_prompt}"
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")
    

def enhance_prompt_with_meta_ai(user_prompt, task_objective, task_description):
    url = "https://meta-ai.example.com/text-enhance"  
    headers = {
        'Authorization': f'Bearer {META_AI_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'prompt': f"{task_objective}. {task_description}. {user_prompt}"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        return response.json().get('enhanced_prompt', '')
    else:
        raise Exception(f"Meta AI API error: {response.status_code} - {response.text}")





def generate_images(service, prompt):
    try:
        if service == 'clipdrop':
            url = 'https://clipdrop-api.co/text-to-image/v1'
            headers = {'x-api-key': CLIPDROP_API_KEY}
            files = {'prompt': (None, prompt)}  
            response = requests.post(url, files=files, headers=headers)
        
            print(f"Clipdrop API Response - Status Code: {response.status_code}")
            print(f"Clipdrop API Response - Body: {response.text}")
            
            if response.ok:
                return base64.b64encode(response.content).decode('utf-8')
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None

        elif service == 'limewire':
            url = "https://api.limewire.com/api/image/generation"
            headers = {
                "Content-Type": "application/json",
                "X-Api-Version": "v1",
                "Accept": "application/json",
                "Authorization": "Bearer YOUR_LIMEWIRE_API_KEY"
            }
            payload = {"prompt": prompt, "aspect_ratio": "1:1"}
            response = requests.post(url, json=payload, headers=headers)
            if response.ok:
                response_data = response.json()
                image_url = response_data.get('image_url')
                if image_url:
                    image_response = requests.get(image_url)
                    if image_response.ok:
                        return base64.b64encode(image_response.content).decode('utf-8')
            return None

        elif service == 'stability_ai':
            url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
            headers = {"authorization": f"Bearer {STABILITY_AI_API_KEY}", "accept": "image/*"}
            data = {"prompt": prompt, "output_format": "jpeg"}
            response = requests.post(url, headers=headers, data=data, files={"none": ""})
            if response.status_code == 200:
                return base64.b64encode(response.content).decode("utf-8")
            else:
                error_message = response.json() if response.headers.get("Content-Type") == "application/json" else response.text
                print(f"Error: {response.status_code} - {error_message}")
                return None

        else:
            return None

    except requests.RequestException as e:
        print(f"RequestException occurred: {str(e)}")
        raise e
    except Exception as e:
        print(f"Unexpected exception: {str(e)}")
        raise e
    


def evaluate_prompt(user_prompt, task_objective, task_description):
    """
    Evaluates the user's prompt using Gemini API and returns a score between 0-100
    """
    evaluation_prompt = f"""Analyze this prompt based on the task requirements and provide a score (0-100) 
    considering these factors:
    1. Relevance to task objective: {task_objective}
    2. Alignment with task description: {task_description}
    3. Creativity and originality
    4. Clarity and specificity
    
    User Prompt: {user_prompt}
    
    Respond ONLY with this format: 
    Score: [number between 0-100]
    Brief Analysis: [2-3 sentence explanation]"""

    try:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}'
        headers = {'Content-Type': 'application/json'}
        data = {'contents': [{'parts': [{'text': evaluation_prompt}]}]}
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        gemini_response = response.json()
        evaluation_text = gemini_response['candidates'][0]['content']['parts'][0]['text']
        
        score_match = re.search(r'Score:\s*(\d+)', evaluation_text)
        analysis_match = re.search(r'Brief Analysis:\s*(.+)', evaluation_text, re.DOTALL)
        
        if score_match:
            score = min(max(int(score_match.group(1)), 0), 100)  
            analysis = analysis_match.group(1).strip() if analysis_match else "No analysis provided"
            return {'score': score, 'analysis': analysis}
        
        return {'score': None, 'analysis': "Could not determine score"}
    
    except Exception as e:
        print(f"Scoring error: {str(e)}")
        return {'score': None, 'analysis': "Scoring system unavailable"}




# Define levels, tasks with objectives and descriptions, and challenges
LEVELS = {
    1: {
        'tasks': [
            {
                'task_number': 1,
                'description': "The ancient city of Lumora, hidden in the heart of the Celestial Valley, is on the brink of collapse. A forbidden artifact known as the Echo Core has been uncovered, pulsing with mysterious energy. The Scholars of Light seek answers, while the Shadowborn plot in the dark. As a Promptian, your first task is to craft a compelling prompt that will guide the Oracle of Lumora to reveal the artifacts true nature.",
                'objective': "Create a precise and effective prompt that asks the Oracle to describe the origins, powers, and possible dangers of the Echo Core."
            },
            {
                'task_number': 2,
                'description': "Illustrate the mystical city of Lumora at twilight.",
                'objective': "Design a cityscape where ancient architecture blends with celestial energy, emphasizing glowing artifacts and floating structures."
            },
            {
                'task_number': 3,
                'description': "Describe a legend about the Echo Core that has been passed down through generations.",
                'objective': "Write a detailed legend explaining the origins of the Echo Core and the consequences of its misuse."
            },
            {
                'task_number': 4,
                'description': "Visualize the Scholars of Light in their grand library.",
                'objective': "Depict a group of wise scholars in a vast, ethereal library filled with floating books and luminous scrolls."
            },
            {
                'task_number': 5,
                'description': "Design a Shadowborn leader plotting in the depths of Lumora.",
                'objective': "Illustrate a sinister and mysterious leader, shrouded in darkness, orchestrating plans to claim the Echo Core."
            }
        ],
        'challenge': "Combine elements from all tasks into a single grand illustration that represents Lumoras unfolding conflict."
    },
    2: {
        'tasks': [
            {
                'task_number': 1,
                'description': "The Echo Cores awakening has triggered celestial disturbances. Strange portals have begun appearing throughout Lumora, revealing glimpses of unknown worlds. The scholars are divided—some see it as an opportunity, others as an omen. Your next task is to craft a prompt for the Oracle that asks about the nature of these portals and where they lead.",
                'objective': "Create a well-structured prompt that elicits a detailed response about the origins, destinations, and effects of these mysterious portals."
            },
            {
                'task_number': 2,
                'description': "Illustrate a celestial portal emerging in the city of Lumora.",
                'objective': "Design a glowing, otherworldly portal tearing through reality, reflecting the mystery and danger it brings."
            },
            {
                'task_number': 3,
                'description': "Write a dialogue between a Scholar of Light and a Shadowborn discussing the portals.",
                'objective': "Create an engaging dialogue that highlights their differing perspectives on the emerging portals and their potential consequences."
            },
            {
                'task_number': 4, 
                'description': "Visualize a new world seen through one of the portals.",
                'objective': "Illustrate an exotic, alien landscape visible through the celestial portal, capturing its uniqueness and mystery."
            },
            {
                'task_number': 5,
                'description': "Depict the citys reaction to the sudden appearance of portals.",
                'objective': "Illustrate the chaos, awe, and fear as the citizens of Lumora witness these cosmic rifts appearing around them."
            }
        ],
        'challenge': "Create a narrative-driven artwork that tells the story of Lumoras portal crisis, combining elements from all tasks."
    },
    3: {
        'tasks': [
            {
                'task_number': 1,
                'description': "The portals have begun to destabilize, and the Echo Core is at risk of shattering, unleashing a force beyond comprehension. The last hope lies in finding the Lost Guardians—ancient beings rumored to hold the key to controlling the artifact s power. Your next task is to craft a prompt that will guide the Oracle in revealing the location and nature of the Lost Guardians.",
                'objective': "Write a compelling prompt that asks the Oracle for specific details about the Lost Guardians, their abilities, and their connection to the Echo Core."
            },
            {
                'task_number': 2,
                'description': "Illustrate one of the Lost Guardians emerging from the void.",
                'objective': "Design a powerful and enigmatic figure, partially shrouded in cosmic energy, embodying ancient wisdom and raw power."
            },
            {
                'task_number': 3,
                'description': "Write a prophecy foretelling the rise of the Lost Guardians.",
                'objective': "Compose a poetic and cryptic prophecy that hints at the return of the Lost Guardians and their role in saving Lumora."
            },
            {
                'task_number': 4,
                'description': "Visualize the final confrontation between the Scholars of Light and the Shadowborn.",
                'objective': "Create an epic battle scene where both factions clash under the glow of the unstable Echo Core, each fighting for control."
            },
            {
                'task_number': 5,
                'description': "Depict the moment when the Echo Core either shatters or is stabilized.",
                'objective': "Illustrate the climactic moment of the story, capturing the energy, destruction, or salvation of Lumora s most powerful artifact."
            }
        ],
        'challenge': "Synthesize all elements into a single, breathtaking illustration that captures the climax of Lumora s fate."
    }
}


@authentication_classes([JWTAuthentication])  
@permission_classes([IsAuthenticated]) 
@csrf_exempt
def make_image(request):
    if request.method == 'POST':
        try:    
            data = json.loads(request.body.decode('utf-8'))
            level_number = data.get('level')
            task_number = data.get('task')
            user_prompt = data.get('prompt')
            service = data.get('service', 'clipdrop')

            # Input validation
            if not all([isinstance(level_number, int), isinstance(task_number, int), user_prompt]):
                return JsonResponse({'error': 'Invalid input parameters'}, status=400)

            # Level/task validation
            level_data = LEVELS.get(level_number)
            if not level_data:
                return JsonResponse({'error': 'Invalid level number'}, status=400)
            
            tasks = level_data.get('tasks', [])
            if not 1 <= task_number <= len(tasks):
                return JsonResponse({'error': 'Invalid task number'}, status=400)

            # task details
            task = tasks[task_number - 1]
            task_obj = task['objective']
            task_desc = task['description']

            # Enhance prompt
            enhanced_prompt = enhance_prompt_with_gemini(user_prompt, task_obj, task_desc)
            
            # Generate score and analysis
            evaluation = evaluate_prompt(user_prompt, task_obj, task_desc)
            
            # Generate image
            image_data = generate_images(service, enhanced_prompt)
            if not image_data:
                return JsonResponse({'error': 'Image generation failed'}, status=500)

            # Save to database
            generated_image = GeneratedImage(
                prompt=enhanced_prompt,
                image_data=image_data,
                user_prompt=user_prompt,
                score=evaluation['score'],
                analysis=evaluation['analysis']
            )
            generated_image.save()

            response_data = {
                'images': [image_data],
                'score_details': {
                    'score': evaluation['score'],
                    'analysis': evaluation['analysis'],
                    'max_score': 100,
                    'criteria': ['Relevance', 'Creativity', 'Clarity', 'Task Alignment']
                },
                'task_progress': {
                    'current_task': task_number,
                    'total_tasks': len(tasks),
                    'next_task': task_number + 1 if task_number < len(tasks) else None
                },
                'enhanced_prompt': enhanced_prompt,
                'challenge': level_data['challenge']
            }

            return JsonResponse(response_data)

        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)
        except Exception as e:
            print(f"System error: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)

    return render(request, 'index.html', {'levels': LEVELS})


TEXT_SERVICES = {
    'gemini': enhance_prompt_with_gemini,
    'openai': enhance_prompt_with_openai,
    'meta_ai': enhance_prompt_with_meta_ai
}


@csrf_exempt
def make_text(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            level_number = data.get('level', None)
            task_number = data.get('task', None)
            user_prompt = data.get('prompt', None)
            service = data.get('service', 'gemini')

            # Validate input
            if level_number is None or task_number is None or user_prompt is None:
                return JsonResponse({'error': 'Level, task number, and prompt are required.'}, status=400)

            # Validate level and task
            if level_number not in LEVELS:
                return JsonResponse({'error': 'Invalid level number.'}, status=400)
            tasks_for_level = LEVELS[level_number].get('tasks', [])
            if task_number <= 0 or task_number > len(tasks_for_level):
                return JsonResponse({'error': 'Invalid task number for the selected level.'}, status=400)

            # Get task details
            selected_task = tasks_for_level[task_number - 1]
            task_description = selected_task['description']
            task_objective = selected_task['objective']

            # Validate service
            if service not in TEXT_SERVICES:
                return JsonResponse({'error': f'Unsupported service: {service}'}, status=400)

            # Enhance the prompt using the selected service
            enhance_function = TEXT_SERVICES[service]
            enhanced_prompt = enhance_function(user_prompt, task_objective, task_description)

            # Evaluate original prompt
            evaluation = evaluate_prompt(user_prompt, task_objective, task_description)

            response_data = {
                'enhanced_prompt': enhanced_prompt,
                'score_details': {
                    'score': evaluation['score'],
                    'analysis': evaluation['analysis'],
                    'max_score': 100,
                    'criteria': [
                        'Relevance to Objective', 
                        'Task Alignment',
                        'Creativity',
                        'Clarity'
                    ]
                },
                'task_progress': {
                    'current_task': task_number,
                    'total_tasks': len(tasks_for_level),
                    'next_task': task_number + 1 if task_number < len(tasks_for_level) else None
                },
                'challenge': LEVELS[level_number]['challenge'],
                'feedback': {
                    'strengths': identify_strengths(user_prompt),  
                    'improvements': suggest_improvements(user_prompt)  
                }
            }

            return JsonResponse(response_data)

        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'Invalid JSON input: {str(e)}'}, status=400)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)

    return render(request, 'index.html', {'levels': LEVELS})

# Helper functions for detailed feedback
def identify_strengths(prompt):
    """Identify key strengths in the user's prompt"""
    analysis_prompt = f"""Analyze this text prompt and identify its main strengths:
    {prompt}
    
    Respond with a bullet list of 3-5 strengths focusing on:
    - Creativity
    - Clarity
    - Originality
    - Technical merit"""
    
    return get_ai_analysis(analysis_prompt)

def suggest_improvements(prompt):
    """Suggest actionable improvements for the prompt"""
    improvement_prompt = f"""Provide constructive feedback to improve this text prompt:
    {prompt}
    
    Give 3-5 concrete suggestions focusing on:
    - Specificity
    - Structure
    - Creativity boost
    - Technical enhancements"""
    
    return get_ai_analysis(improvement_prompt)

def get_ai_analysis(prompt):
    """Generic helper for Gemini analysis"""
    try:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}'
        headers = {'Content-Type': 'application/json'}
        data = {'contents': [{'parts': [{'text': prompt}]}]}
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return "Feedback generation unavailable"


from django.contrib.auth import get_user_model
import traceback

User = get_user_model()  # Ensure we get the correct User model

@login_required
def get_user_scores(request):
    try:
        # Ensure request.user is authenticated
        if not request.user.is_authenticated:
            return JsonResponse({"error": "User not authenticated."}, status=401)

        # If request.user is a string (username), convert it to a User instance
        if isinstance(request.user, str):
            try:
                user = User.objects.get(username=request.user)
            except User.DoesNotExist:
                return JsonResponse({"error": "User not found."}, status=404)
        else:
            user = request.user  # User is already an instance

        # Fetch user progress, create if doesn't exist
        progress, created = UserProgress.objects.get_or_create(user=user)

        return JsonResponse({
            "total_score": progress.total_score,
            "completed_tasks": progress.completed_tasks
        })

    except Exception as e:
        error_message = str(e)
        traceback.print_exc()  # Print error details in the terminal
        return JsonResponse({"error": error_message}, status=500)
    
from django.http import JsonResponse
    
def get_image_scores(request):
    """
    Retrieve scores and details for all generated images.
    """
    try:
        # Fetch all generated images with their scores and analysis
        images = GeneratedImage.objects.all().values('id', 'user__username', 'prompt', 'score', 'analysis', 'created_at')
        
        # Convert QuerySet to list for JSON serialization
        image_scores = list(images)
        
        return JsonResponse({'image_scores': image_scores}, status=200)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def get_text_scores(request):
    """
    Retrieve scores and details for all text prompts and feedback.
    """
    try:
        # Fetch all prompt feedback with their scores and analysis
        feedbacks = PromptFeedback.objects.all().values('id', 'user__username', 'original_prompt', 'enhanced_prompt', 'score', 'analysis', 'created_at')
        
        # Convert QuerySet to list for JSON serialization
        text_scores = list(feedbacks)
        
        return JsonResponse({'text_scores': text_scores}, status=200)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)