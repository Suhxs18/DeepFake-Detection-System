from django.urls import path
from . import views

urlpatterns = [
    # Authentication routes
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    
    # Profile routes
    path('profile/create/', views.create_profile, name='create_profile'),
    path('profile/update/', views.update_profile, name='update_profile'),
    path('profile/view/', views.view_profile, name='view_profile'),
    
    # Dashboard and advice routes
    path('dashboard/', views.dashboard, name='dashboard'),
    path('advice/new/', views.investment_advice, name='new_advice'),
    path('advice/history/', views.advice_history, name='advice_history'),
    path('advice/<int:advice_id>/', views.view_advice, name='view_advice'),
    
    # API routes
    path('api/chat/', views.chat_with_advisor, name='chat_api'),
    path('api/chat/history/', views.get_chat_history, name='chat_history_api'),
]
