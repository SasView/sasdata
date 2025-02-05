from django.urls import path
from dj_rest_auth.views import (LoginView, LogoutView,
                                UserDetailsView, PasswordChangeView)
from dj_rest_auth.registration.views import RegisterView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('user/', UserDetailsView.as_view(), name='view user information'),
    path('password/change/', PasswordChangeView.as_view(), name='change password'),
]