from django.shortcuts import render
from django.conf import settings
# Create your views here.
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework import status
from .models import FileModel, AlgorithmModel, AnalyticModel
from .serializers import FileSerializer,AlgorithmSerializer, AnalyticSerializer
import os

def get_file_list():
    files=FileModel.objects.values_list('file_name',flat=True)
    return files.distinct()

def get_algorithm_list():
    algorithmName = AlgorithmModel.objects.values_list('algorithm_name',flat=True)
    return algorithmName.distinct()
    # return ['PimaDiabetesClassifier','WineQualityRegressor','NBAOutlier']

def get_algorithm(name):
    return AlgorithmModel.objects.get(algorithm_name = name)
    # if name == 'PimaDiabetesClassifier':
    #     from .classifier import run_algo1
    #     return run_algo1
    # elif name == 'WineQualityRegressor':
    #     from .regression import run_algo1
    #     return run_algo1
    # elif name == 'NBAOutlier':
    #     from .outlier import run_algo1
    #     return run_algo1
    # else:
    #     # Http404 is imported from django.http
    #     raise Http404('<h1>Target algorithm not found</h1>')


def run_analytic(file_path, algo_object):
    # file_abs_path = os.path.join(settings.MEDIA_ROOT, str(file_path))
    # return algo_object(file_abs_path)
    algo_model = algo_object.saved_model
    algo_script = algo_object.inference_script
    file_abs_path = os.path.join(settings.MEDIA_ROOT, str(file_path))
    algo_model_path = os.path.join(settings.MEDIA_ROOT, str(algo_model))
    algo_model_import_path = str(algo_script).\
    replace('/','.').replace('\\', '.')[:-3]
    algo_model_import_path = 'media.'+algo_model_import_path
    print('Running algorithm from: ', algo_model_import_path)
    algo_model = __import__(algo_model_import_path, fromlist=[algo_model_import_path])
    return algo_model.run_algo1(file_abs_path, algo_model_path)

class YourViewName(APIView):
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    renderer_classes = [TemplateHTMLRenderer]
    template_name = 'index.html'
    # See Django REST Request class here:
    # https://www.django-rest-framework.org/api-guide/requests/
    
    def get(self, request):
        f = FileModel.objects.all()
        a = AnalyticModel.objects.all()     
        #l = len(f)
        return Response({'f': f, 'l' : len(f), 'files': get_file_list(),
                                'algorithms':get_algorithm_list(),'a':a},status=status.HTTP_200_OK)

    def post(self, request):
        # Upload form
        if 'upload' in request.data:
            file_serializer = FileSerializer(data=request.data)
            if file_serializer.is_valid():
                data = (file_serializer.validated_data)['file_content']
            name = data.name
            f = FileModel.objects.all()
            if(name.find('.csv') == -1):
                return Response({'status': 'Wrong File Type. Upload only .csv files','f':f, 'l' : len(f), 'files': get_file_list(),
                                'algorithms':get_algorithm_list()},status=status.HTTP_201_CREATED)
            else:    
                file_serializer.save()
                return Response({'status': 'Upload successful!','f':f, 'l' : len(f), 'files': get_file_list(),
                                'algorithms':get_algorithm_list()},status=status.HTTP_201_CREATED)
       
        elif 'delete' in request.data:
            file_id = request.data['delete']
            fdel = FileModel.objects.get(pk=file_id)
            fdel.file_content.delete()
            fdel.delete()
            f = FileModel.objects.all()
            return Response({'f':f, 'l' : len(f), 'files': get_file_list(),
                                'algorithms':get_algorithm_list()},status=status.HTTP_200_OK)

        elif 'analytic' in request.data:
            f = FileModel.objects.all()
            # Run analytics on dataset as specified by file_name and
            # analytic_id received in the post request
            query_file_name = request.data['file_name']
            query_algorithm = request.data['algorithm']
            # Find file path to local folder
            file_obj = FileModel.objects.get(file_name=query_file_name)
            file_path = file_obj.file_content
            # Find algorithm
            algo_obj = get_algorithm(query_algorithm)
            analyticresult = run_analytic(file_path, algo_obj)

            to_save = {'analytic_name': query_file_name+'_'+query_algorithm,
                        'result_plot': analyticresult}#,'DataSet_name':query_file_name,'Algo_name':query_algorithm}
            analytic_serializer = AnalyticSerializer(data=to_save)
            if analytic_serializer.is_valid():
                analytic_serializer.save()
                # The return statement as before
            else:
                # HttpResponse is imported from django.http
                return HttpResponse('The server encountered an internal error'
                +'while processing'+to_save['analytic_name'],
                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({'files':get_file_list(),
            'algorithms':get_algorithm_list(),
            'result_plot':analyticresult,
            'f': f, 'l' : len(f)},
            status=status.HTTP_200_OK)

        else:
            f = FileModel.objects.all()
            return Response({'f':f, 'l' : len(f), 'files': get_file_list(),
                            'algorithms':get_algorithm_list()},status=status.HTTP_400_BAD_REQUEST)

                            
