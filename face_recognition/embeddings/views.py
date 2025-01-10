from django.shortcuts import render
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from .models import FaceEmbedding
from insightface.app import FaceAnalysis
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import FaceEmbedding

# Initialize FaceAnalysis (ensure the model files are properly placed)
faceapp = FaceAnalysis(name='buffalo_l', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

def register(request):
    if request.method == 'POST':
        person_name = request.POST.get('person_name')
        role = request.POST.get('role')

        if not person_name or not role:
            return JsonResponse({'error': 'Name and role are required'}, status=400)

        key = f"{person_name}@{role}"
        print(f"Registering {person_name} as {role} with key {key}")

        # Initialize the video capture object
        cap = cv2.VideoCapture(0)
        face_embeddings = []
        sample = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Unable to read from camera')
                return JsonResponse({'error': 'Unable to access the camera'}, status=500)

            # Perform face detection using insightface
            results = faceapp.get(frame, max_num=1)

            for res in results:
                sample += 1
                # Extract bounding box and draw rectangle
                x1, y1, x2, y2 = res['bbox'].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 206, 135), 2)

                # Extract facial features (embedding)
                embeddings = res['embedding']
                face_embeddings.append(embeddings)

            # Stop collecting embeddings after 200 samples
            if sample >= 200:
                break

            # Display the frame
            cv2.imshow('frame', frame)

            # Break on 'q' key press
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Calculate the mean embedding
        x_mean = np.asarray(face_embeddings).mean(axis=0)
        x_mean_bytes = x_mean.tobytes()

        # Save data to the database
        FaceEmbedding.objects.create(
            person_name=person_name,
            role=role,
            embedding=x_mean_bytes
        )

        return JsonResponse({'success': f"Registered {person_name} successfully"})

    return render(request, 'register.html')




class RegisterAPI(APIView):
    def post(self, request):
        person_name = request.data.get('person_name')
        role = request.data.get('role')

        # Perform face detection and save embeddings (similar to register view)
        # ...
        
        return Response({'message': 'User registered successfully!'}, status=status.HTTP_201_CREATED)

def login(request):
    if request.method == 'POST':
        # Initialize camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        if not ret:
            return JsonResponse({'error': 'Unable to access the camera'}, status=500)

        # Perform face detection
        results = faceapp.get(frame, max_num=1)
        cap.release()
        cv2.destroyAllWindows()

        if not results:
            return JsonResponse({'error': 'No face detected'}, status=400)

        # Extract embedding from the detected face
        live_embedding = results[0]['embedding']

        # Compare with embeddings in the database
        all_embeddings = FaceEmbedding.objects.all()
        for record in all_embeddings:
            stored_embedding = np.frombuffer(record.embedding, dtype=np.float32)
            similarity = cosine_similarity([live_embedding], [stored_embedding])[0][0]

            if similarity > 0.8:  # Threshold for similarity
                return JsonResponse({'success': f"Welcome back, {record.person_name}!"})

        return JsonResponse({'error': 'Face not recognized'}, status=401)

    return render(request, 'login.html')
