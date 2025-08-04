from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from django.contrib.auth.models import User


# Create your tests here.
class AuthTests(TestCase):
    """Tests for authentication endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.client1 = APIClient()
        cls.client2 = APIClient()
        cls.register_data = {
            "email": "email@domain.org",
            "username": "testUser",
            "password1": "sasview!",
            "password2": "sasview!",
        }
        cls.login_data = {
            "username": "testUser",
            "email": "email@domain.org",
            "password": "sasview!",
        }
        cls.login_data_2 = {
            "username": "testUser2",
            "email": "email2@domain.org",
            "password": "sasview!",
        }
        cls.user = User.objects.create_user(
            id=1, username="testUser2", password="sasview!", email="email2@domain.org"
        )
        cls.client_authenticated = APIClient()
        cls.client_authenticated.force_authenticate(user=cls.user)

    # Create an authentication header for a given token
    def auth_header(self, response):
        return {"Authorization": "Token " + response.data["token"]}

    # Test if registration successfully creates a new user and logs in
    def test_register(self):
        response = self.client1.post("/auth/register/", data=self.register_data)
        user = User.objects.get(username="testUser")
        response2 = self.client1.get("/auth/user/", headers=self.auth_header(response))
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(user.email, self.register_data["email"])
        self.assertEqual(response2.status_code, status.HTTP_200_OK)
        user.delete()

    # Test if login successful
    def test_login(self):
        response = self.client1.post("/auth/login/", data=self.login_data_2)
        response2 = self.client1.get("/auth/user/", headers=self.auth_header(response))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)

    # Test simultaneous login by multiple clients
    def test_multiple_login(self):
        response = self.client1.post("/auth/login/", data=self.login_data_2)
        response2 = self.client2.post("/auth/login/", data=self.login_data_2)
        response3 = self.client1.get("/auth/user/", headers=self.auth_header(response))
        response4 = self.client2.get("/auth/user/", headers=self.auth_header(response2))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)
        self.assertEqual(response3.status_code, status.HTTP_200_OK)
        self.assertEqual(response4.status_code, status.HTTP_200_OK)
        self.assertNotEqual(response.content, response2.content)

    # Test get user information
    def test_user_get(self):
        response = self.client_authenticated.get("/auth/user/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.content,
            b'{"pk":1,"username":"testUser2","email":"email2@domain.org","first_name":"","last_name":""}',
        )

    # Test changing username
    def test_user_put_username(self):
        data = {"username": "newName"}
        response = self.client_authenticated.put("/auth/user/", data=data)
        self.user.username = "testUser2"
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.content,
            b'{"pk":1,"username":"newName","email":"email2@domain.org","first_name":"","last_name":""}',
        )

    # Test changing username and first and last name
    def test_user_put_name(self):
        data = {"username": "newName", "first_name": "Clark", "last_name": "Kent"}
        response = self.client_authenticated.put("/auth/user/", data=data)
        self.user.first_name = ""
        self.user.last_name = ""
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.content,
            b'{"pk":1,"username":"newName","email":"email2@domain.org","first_name":"Clark","last_name":"Kent"}',
        )

    # Test user info inaccessible when unauthenticated
    def test_user_unauthenticated(self):
        response = self.client1.get("/auth/user/")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(
            response.content,
            b'{"detail":"Authentication credentials were not provided."}',
        )

    # Test logout is successful after login
    def test_login_logout(self):
        self.client1.post("/auth/login/", data=self.login_data_2)
        response = self.client1.post("/auth/logout/")
        response2 = self.client1.get("/auth/user/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content, b'{"detail":"Successfully logged out."}')
        self.assertEqual(response2.status_code, status.HTTP_401_UNAUTHORIZED)

    # Test logout is successful after registration
    def test_register_logout(self):
        self.client1.post("/auth/register/", data=self.register_data)
        response = self.client1.post("/auth/logout/")
        response2 = self.client1.get("/auth/user/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content, b'{"detail":"Successfully logged out."}')
        self.assertEqual(response2.status_code, status.HTTP_401_UNAUTHORIZED)
        User.objects.get(username="testUser").delete()

    # Test multiple logins for the same account log out independently
    def test_multiple_logout(self):
        self.client1.post("/auth/login/", data=self.login_data_2)
        token = self.client2.post("/auth/login/", data=self.login_data_2)
        response = self.client1.post("/auth/logout/")
        response2 = self.client2.get("/auth/user/", headers=self.auth_header(token))
        response3 = self.client2.post("/auth/logout/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)
        self.assertEqual(response3.status_code, status.HTTP_200_OK)

    # Test login is successful after registering then logging out
    def test_register_login(self):
        register_response = self.client1.post(
            "/auth/register/", data=self.register_data
        )
        logout_response = self.client1.post("/auth/logout/")
        login_response = self.client1.post("/auth/login/", data=self.login_data)
        self.assertEqual(register_response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(logout_response.status_code, status.HTTP_200_OK)
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)
        User.objects.get(username="testUser").delete()

    # Test password is successfully changed
    def test_password_change(self):
        data = {
            "new_password1": "sasview?",
            "new_password2": "sasview?",
            "old_password": "sasview!",
        }
        self.login_data_2["password"] = "sasview?"
        response = self.client_authenticated.post("/auth/password/change/", data=data)
        login_response = self.client1.post("/auth/login/", data=self.login_data_2)
        self.login_data_2["password"] = "sasview!"
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)

    @classmethod
    def tearDownClass(cls):
        cls.user.delete()
