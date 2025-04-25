from knox.models import AuthToken


# create an authentication token
def create_knox_token(token_model, user, serializer):
    token = AuthToken.objects.create(user=user)
    return token
