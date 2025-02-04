import requests

from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from django.contrib.auth.models import User

# Create your tests here.
class AuthTests(TestCase):

    def setUp(self):
        self.client = APIClient()
        self.register_data = {
            "email": "email@domain.org",
            "username": "testUser",
            "password1": "sasview!",
            "password2": "sasview!"
        }
        self.login_data = {
            "username": "testUser",
            "email": "email@domain.org",
            "password": "sasview!"
        }

    def test_register(self):
        response = self.client.post('/dj-rest-auth/registration/',data=self.register_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        user = User.objects.get(username="testUser")
        self.assertEquals(user.email, self.register_data["email"])

    def test_login(self):
        user = User.objects.create_user(username="testUser", password="sasview!", email="email@domain.org")
        response = self.client.post('/dj-rest-auth/login', data=self.login_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_login_logout(self):
        user = User.objects.create_user(username="testUser", password="sasview!", email="email@domain.org")
        self.client.post('/dj-rest-auth/login', data=self.login_data)
        response = self.client.post('/dj-rest-auth/logout')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content, b'{"detail":"Successfully logged out."}')

    def test_register_logout(self):
        self.client.post('/dj-rest-auth/registration/', data=self.register_data)
        response = self.client.post('/dj-rest-auth/logout')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content, b'{"detail":"Successfully logged out."}')

    def test_register_login(self):
        register_response = self.client.post('/dj-rest-auth/registration/', data=self.register_data)
        logout_response = self.client.post('/dj-rest-auth/logout')
        login_response = self.client.post('/dj-rest-auth/login', data=self.login_data)
        self.assertEqual(register_response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(logout_response.status_code, status.HTTP_200_OK)
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)

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

