from django.contrib import admin
from django.urls import path, include, re_path
from django.views.decorators.csrf import csrf_exempt

# for React templates
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('API.urls'))
]
urlpatterns += [re_path(r"^.*",
                        csrf_exempt(TemplateView.as_view(template_name="index.html")))]
