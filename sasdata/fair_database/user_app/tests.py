import requests

from django.test import TestCase
from rest_framework import status
from rest_framework.test import RequestsClient

# Create your tests here.
class AuthTests(TestCase):

    def setup(self):
        self.client = RequestsClient()

    def test_register(self):
        data = {
            'email': 'test@test.com',
            'username': 'testUser',
            'password': 'testPassword'
        }
        response = self.client.post('/_allauth/app/v1/auth/signup',data=data)
        print(response.content)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

#can register a user, user is w/in User model
# user is logged in after registration
# logged-in user can create Data, is data's current_user
# test log out


# Permissions
# Any user can access public data
# logged-in user can access and modify their own private data
# unauthenticated user cannot access private data
# unauthenticated user cannot modify data
# logged-in user cannot modify data other than their own
# logged-in user cannot access the private data of others

