from django.urls import path
from dj_rest_auth.views import (LogoutView,
                                UserDetailsView, PasswordChangeView)
from .views import KnoxLoginView, KnoxRegisterView, OrcidLoginView

'''Urls for authentication. Orcid login not functional.'''

urlpatterns = [
    path('register/', KnoxRegisterView.as_view(), name='register'),
    path('login/', KnoxLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('user/', UserDetailsView.as_view(), name='view user information'),
    path('password/change/', PasswordChangeView.as_view(), name='change password'),
    path('login/orcid/', OrcidLoginView.as_view(), name='orcid login')
]