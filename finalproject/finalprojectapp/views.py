from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework import status
from .models import FileModel
from .serializers import FileSerializer
class YourViewName(APIView):
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    renderer_classes = [TemplateHTMLRenderer]
    template_name = 'index.html'
    # See Django REST Request class here:
    # https://www.django-rest-framework.org/api-guide/requests/
    def get(self, request):
        return Response(status=status.HTTP_200_OK)
    def post(self, request):
        # Upload form
        if 'upload' in request.data:
            file_serializer = FileSerializer(data=request.data)
        if file_serializer.is_valid():
            data = (file_serializer.validated_data)['file_content']
            name = data.name
            if(name.find('.csv') == -1):
                return Response({'status': 'Wrong File Type. Upload only .csv files'},status=status.HTTP_201_CREATED)
            else:
                file_serializer.save()
                return Response({'status': 'Upload successful!'},status=status.HTTP_201_CREATED)
        else:
            return Response(status=status.HTTP_400_BAD_REQUEST)