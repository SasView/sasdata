from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

from data.models import PublishedState, Session


class TestSessionWithPublishedState(APITestCase):
    @classmethod
    def setUpTestData(cls):
        pass

    # Test creating a session with a published state
    # Should this just be a part of sessions testing?

    # Test GET on a session with a published state

    # Test PUT on nested published state

    # Test cascading delete

    @classmethod
    def tearDownClass(cls):
        pass


class TestPublishedStateView(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user1 = User.objects.create_user(
            id=1, username="testUser1", password="secret"
        )
        cls.user2 = User.objects.create_user(
            id=2, username="testUser2", password="secret"
        )
        cls.public_session = Session.objects.create(
            id=1, current_user=cls.user1, title="Public Session", is_public=True
        )
        cls.private_session = Session.objects.create(
            id=2, current_user=cls.user1, title="Private Session", is_public=False
        )
        cls.unowned_session = Session.objects.create(
            id=3, title="Unowned Session", is_public=True
        )
        cls.public_ps = PublishedState.objects.create(
            id=1,
            doi="http://127.0.0.1:8000/v1/data/session/1/",
            published=True,
            session=cls.public_session,
        )
        cls.private_ps = PublishedState.objects.create(
            id=2,
            doi="http://127.0.0.1:8000/v1/data/session/2/",
            published=False,
            session=cls.private_session,
        )
        cls.unowned_ps = PublishedState.objects.create(
            id=3,
            doi="http://127.0.0.1:8000/v1/data/session/3/",
            published=True,
            session=cls.unowned_session,
        )
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)

    # Test listing published states - various permissions
    def test_list_published_states_private(self):
        request = self.auth_client1.get("/v1/data/published/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": "http://127.0.0.1:8000/v1/data/session/1/",
                    },
                    2: {
                        "title": "Private Session",
                        "published": False,
                        "doi": "http://127.0.0.1:8000/v1/data/session/2/",
                    },
                    3: {
                        "title": "Unowned Session",
                        "published": True,
                        "doi": "http://127.0.0.1:8000/v1/data/session/3/",
                    },
                }
            },
        )

    def test_list_published_states_public(self):
        request = self.auth_client2.get("/v1/data/published/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": "http://127.0.0.1:8000/v1/data/session/1/",
                    },
                    3: {
                        "title": "Unowned Session",
                        "published": True,
                        "doi": "http://127.0.0.1:8000/v1/data/session/3/",
                    },
                }
            },
        )

    def test_list_published_states_shared(self):
        self.private_session.users.add(self.user2)
        request = self.auth_client2.get("/v1/data/published/")
        self.private_session.users.remove(self.user2)
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": "http://127.0.0.1:8000/v1/data/session/1/",
                    },
                    2: {
                        "title": "Private Session",
                        "published": False,
                        "doi": "http://127.0.0.1:8000/v1/data/session/2/",
                    },
                    3: {
                        "title": "Unowned Session",
                        "published": True,
                        "doi": "http://127.0.0.1:8000/v1/data/session/3/",
                    },
                }
            },
        )

    def test_list_published_states_unauthenticated(self):
        request = self.client.get("/v1/data/published/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": "http://127.0.0.1:8000/v1/data/session/1/",
                    },
                    3: {
                        "title": "Unowned Session",
                        "published": True,
                        "doi": "http://127.0.0.1:8000/v1/data/session/3/",
                    },
                }
            },
        )

    # Test creating a published state

    # Test can only create a published state for your own sessions

    # Test can't create a second published state for a session

    @classmethod
    def tearDownClass(cls):
        cls.public_session.delete()
        cls.private_session.delete()
        cls.unowned_session.delete()
        cls.user1.delete()
        cls.user2.delete()


class TestSinglePublishedStateView(APITestCase):
    @classmethod
    def setUpTestData(cls):
        pass

    # Test viewing a published state - various permissions

    # Test updating a published state

    # Test deleting a published state - session not deleted

    @classmethod
    def tearDownClass(cls):
        pass
