from django.db.models.signals import post_delete
from django.dispatch import receiver

from data.models import OperationTree, Quantity


# TODO: is there a better way to do this than with signals
# delete the operation tree when a quantity is deleted
# see apps.py for signal connection
@receiver(post_delete, sender=Quantity)
def delete_operation_tree(sender, **kwargs):
    if kwargs["instance"].operation_tree:
        kwargs["instance"].operation_tree.delete()


# propagate deletion through the operation tree
# see apps.py for signal connection
@receiver(post_delete, sender=OperationTree)
def delete_parent_operations(sender, **kwargs):
    if kwargs["instance"].parent_operation1:
        kwargs["instance"].parent_operation1.delete()
    if kwargs["instance"].parent_operation2:
        kwargs["instance"].parent_operation2.delete()
