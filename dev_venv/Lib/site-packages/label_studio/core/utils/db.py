import logging
import time
from typing import Optional, TypeVar

from django.db import OperationalError, models, transaction
from django.db.models import Model, QuerySet, Subquery

logger = logging.getLogger(__name__)


class SQCount(Subquery):
    template = '(SELECT count(*) FROM (%(subquery)s) _count)'
    output_field = models.IntegerField()


ModelType = TypeVar('ModelType', bound=Model)


def fast_first(queryset: QuerySet[ModelType]) -> Optional[ModelType]:
    """Replacement for queryset.first() when you don't need ordering,
    queryset.first() works slowly in some cases
    """

    if result := queryset[:1]:
        return result[0]
    return None


def fast_first_or_create(model, **model_params) -> Optional[ModelType]:
    """Like get_or_create, but using fast_first instead of first(). Additionally, unlike get_or_create, this method will not raise an exception if more than one model instance matching the given params is returned, making it a safer choice than get_or_create for models that don't have a uniqueness constraint on the fields used."""
    if instance := fast_first(model.objects.filter(**model_params)):
        return instance
    return model.objects.create(**model_params)


def batch_update_with_retry(queryset, batch_size=500, max_retries=3, **update_fields):
    """
    Update objects in batches with retry logic to handle deadlocks.

    Args:
        queryset: QuerySet of objects to update
        batch_size: Number of objects to update in each batch
        max_retries: Maximum number of retry attempts for each batch
        **update_fields: Fields to update (e.g., overlap=1)
    """
    object_ids = list(queryset.values_list('id', flat=True))
    total_objects = len(object_ids)

    for i in range(0, total_objects, batch_size):
        batch_ids = object_ids[i : i + batch_size]
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                with transaction.atomic():
                    queryset.model.objects.filter(id__in=batch_ids).update(**update_fields)
                break
            except OperationalError as e:
                last_error = e
                if 'deadlock detected' in str(e):
                    retry_count += 1
                    wait_time = 0.1 * (2**retry_count)  # Exponential backoff
                    logger.warning(
                        f'Deadlock detected, retry {retry_count}/{max_retries} '
                        f'for batch {i}-{i+len(batch_ids)}. Waiting {wait_time}s...'
                    )
                    time.sleep(wait_time)
                else:
                    raise
        else:
            logger.error(f'Failed to update batch after {max_retries} retries. ' f'Batch: {i}-{i+len(batch_ids)}')
            raise last_error
