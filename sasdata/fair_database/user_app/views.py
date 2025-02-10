from django.conf import settings

from rest_framework.response import Response
from dj_rest_auth.views import LoginView
from dj_rest_auth.registration.views import RegisterView, SocialLoginView
from allauth.account.utils import complete_signup
from allauth.account import app_settings as allauth_settings
from allauth.socialaccount.providers.orcid.views import OrcidOAuth2Adapter

from user_app.serializers import KnoxSerializer
from user_app.util import create_knox_token

#Login using knox tokens rather than django-rest-framework tokens.

class KnoxLoginView(LoginView):

    def get_response(self):
        serializer_class = self.get_response_serializer()

        data = {
            'user': self.user,
            'token': self.token
        }
        serializer = serializer_class(instance=data, context={'request': self.request})

        return Response(serializer.data, status=200)

# Registration using knox tokens rather than django-rest-framework tokens.
class KnoxRegisterView(RegisterView):

    def get_response_data(self, user):
        return KnoxSerializer({'user': user, 'token': self.token}).data

    def perform_create(self, serializer):
        user = serializer.save(self.request)
        self.token = create_knox_token(None,user,None)
        complete_signup(self.request._request, user, allauth_settings.EMAIL_VERIFICATION, None)
        return user

# For ORCID login
class OrcidLoginView(SocialLoginView):
    adapter_class = OrcidOAuth2Adapter