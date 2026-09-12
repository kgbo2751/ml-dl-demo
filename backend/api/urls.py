from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^predict/tf/?$', views.predict_tf),
    re_path(r'^predict/torch/?$', views.predict_torch),
    re_path(r'^predict/box-office-3d/?$', views.predict_box_office_3d),
    re_path(r'^rating/status/?$', views.get_rating_status),
    re_path(r'^rating/add/?$', views.add_rating),
    re_path(r'^youtube/filter/?$', views.filter_youtube_videos),
]