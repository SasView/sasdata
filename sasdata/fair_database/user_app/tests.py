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

    '''
    def tearDown(self):
        self.client.post('/auth/logout/') '''

    # Test if registration successfully creates a new user and logs in
    def test_register(self):
        response = self.client.post('/auth/register/',data=self.register_data)
        user = User.objects.get(username="testUser")
        response2 = self.client.get('/auth/user/')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(user.email, self.register_data["email"])
        self.assertEqual(response2.status_code, status.HTTP_200_OK)

    # Test if login successful
    def test_login(self):
        User.objects.create_user(username="testUser", password="sasview!", email="email@domain.org")
        response = self.client.post('/auth/login/', data=self.login_data)
        response2 = self.client.get('/auth/user/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)

    # Test get user information
    def test_user_get(self):
        user = User.objects.create_user(username="testUser", password="sasview!", email="email@domain.org")
        self.client.force_authenticate(user=user)
        response = self.client.get('/auth/user/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content,
            b'{"pk":1,"username":"testUser","email":"email@domain.org","first_name":"","last_name":""}')

    # Test changing username
    def test_user_put_username(self):
        user = User.objects.create_user(username="testUser", password="sasview!", email="email@domain.org")
        self.client.force_authenticate(user=user)
        data = {
            "username": "newName"
        }
        response = self.client.put('/auth/user/', data=data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content,
            b'{"pk":1,"username":"newName","email":"email@domain.org","first_name":"","last_name":""}')

    # Test changing username and first and last name
    def test_user_put_name(self):
        user = User.objects.create_user(username="testUser", password="sasview!", email="email@domain.org")
        self.client.force_authenticate(user=user)
        data = {
            "username": "newName",
            "first_name": "Clark",
            "last_name": "Kent"
        }
        response = self.client.put('/auth/user/', data=data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content,
            b'{"pk":1,"username":"newName","email":"email@domain.org","first_name":"Clark","last_name":"Kent"}')

    # Test user info inaccessible when unauthenticated
    def test_user_unauthenticated(self):
        response = self.client.get('/auth/user/')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response.content,
            b'{"detail":"Authentication credentials were not provided."}')

    # Test logout is successful after login
    def test_login_logout(self):
        User.objects.create_user(username="testUser", password="sasview!", email="email@domain.org")
        self.client.post('/auth/login/', data=self.login_data)
        response = self.client.post('/auth/logout/')
        response2 = self.client.get('/auth/user/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content, b'{"detail":"Successfully logged out."}')
        self.assertEqual(response2.status_code, status.HTTP_403_FORBIDDEN)

    # Test logout is successful after registration
    def test_register_logout(self):
        self.client.post('/auth/register/', data=self.register_data)
        response = self.client.post('/auth/logout/')
        response2 = self.client.get('/auth/user/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content, b'{"detail":"Successfully logged out."}')
        self.assertEqual(response2.status_code, status.HTTP_403_FORBIDDEN)

    # Test login is successful after registering then logging out
    def test_register_login(self):
        register_response = self.client.post('/auth/register/', data=self.register_data)
        logout_response = self.client.post('/auth/logout/')
        login_response = self.client.post('/auth/login/', data=self.login_data)
        self.assertEqual(register_response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(logout_response.status_code, status.HTTP_200_OK)
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)

    # Test password is successfully changed
    def test_password_change(self):
        self.client.post('/auth/register/', data=self.register_data)
        data = {
            "new_password1": "sasview?",
            "new_password2": "sasview?",
            "old_password": "sasview!"
        }
        l_data = self.login_data
        l_data["password"] = "sasview?"
        response = self.client.post('/auth/password/change/', data=data)
        logout_response = self.client.post('/auth/logout/')
        login_response = self.client.post('/auth/login/', data=l_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(logout_response.status_code, status.HTTP_200_OK)
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)


# logged-in user can create Data, is data's current_user


# Permissions
# Any user can access public data
# logged-in user can access and modify their own private data
# unauthenticated user cannot access private data
# unauthenticated user cannot modify data
# logged-in user cannot modify data other than their own
# logged-in user cannot access the private data of others

