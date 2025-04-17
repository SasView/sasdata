from rest_framework.test import APITestCase


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
        pass

    # Test listing published states - various permissions

    # Test creating a published state

    # Test can only create a published state for your own sessions

    # Test can't create a second published state for a session

    @classmethod
    def tearDownClass(cls):
        pass


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
