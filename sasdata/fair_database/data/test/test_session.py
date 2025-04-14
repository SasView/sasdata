from rest_framework.test import APITestCase


class TestSession(APITestCase):
    @classmethod
    def setUpTestData(cls):
        pass

    # Test listing sessions
    # Can see own private
    # Can't see other private
    # Can see private with access
    # Can see public (authenticated and unauthenticated)

    # Test listing sessions by username

    # Test creating a session - public, private, unauthenticated
    # Datasets have same access as session

    # Test post fails with dataset validation issue

    @classmethod
    def tearDownClass(cls):
        pass


class TestSingleSession(APITestCase):
    @classmethod
    def setUpTestData(cls):
        pass

    # Test accessing session
    # public, private, with/without access

    # Test updating session
    # public, private, owner, not owner

    # Test deleting session
    # Test delete cascades

    @classmethod
    def tearDownClass(cls):
        pass


class TestSessionAccessManagement(APITestCase):
    @classmethod
    def setUpTestData(cls):
        pass

    # Test listing access

    # Test listing access not as the owner

    # Test granting access

    # Test revoking access

    # Test can't revoke own access

    # Test only owner can change access

    @classmethod
    def tearDownClass(cls):
        pass
