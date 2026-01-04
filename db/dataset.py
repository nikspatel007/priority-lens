#!/usr/bin/env python3
"""PyTorch Dataset backed by SurrealDB.

Provides SurrealEmailDataset as an alternative to the JSON-based EmailDataset,
with efficient querying and optional caching of embeddings in the database.

Usage:
    # Start SurrealDB:
    surreal start file:data/enron.db --user root --pass root

    # Use dataset:
    from db.dataset import create_surreal_dataloaders
    train, val, test = create_surreal_dataloaders(database='enron')
"""

import asyncio
from typing import Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

from surrealdb import AsyncSurreal

# Import feature extraction from main source
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import CombinedFeatureExtractor, FEATURE_DIMS
from src.dataset import ACTION_TO_IDX, IDX_TO_ACTION, NUM_ACTIONS


class SurrealEmailDataset(Dataset):
    """PyTorch Dataset backed by SurrealDB.

    Loads emails from SurrealDB, extracting features on demand or using
    cached embeddings stored in the database.

    Advantages over JSON-based dataset:
    - Efficient querying for specific subsets
    - Graph queries for communication patterns
    - Cached embeddings stored in DB (no recomputation)
    - Streaming for very large datasets

    Example:
        >>> dataset = SurrealEmailDataset(
        ...     database='enron',
        ...     split='train',
        ...     use_cached_embeddings=True
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for features, labels in loader:
        ...     output = model(features)
    """

    def __init__(
        self,
        database: str = 'enron',
        split: str = 'train',
        *,
        url: str = 'ws://localhost:8000/rpc',
        namespace: str = 'rl_emails',
        username: str = 'root',
        password: str = 'root',
        user_email: str = '',
        user_context: Optional[dict] = None,
        use_cached_embeddings: bool = True,
        include_content: bool = False,
        transform: Optional[callable] = None,
    ):
        """Initialize SurrealEmailDataset.

        Args:
            database: Database name ('enron' or 'gmail')
            split: Data split ('train', 'val', 'test', or None for all)
            url: SurrealDB connection URL
            namespace: SurrealDB namespace
            username: SurrealDB username
            password: SurrealDB password
            user_email: Optional user email for feature extraction context
            user_context: Optional historical context for feature extraction
            use_cached_embeddings: If True, use embeddings cached in DB
            include_content: If True, include content embeddings in features
            transform: Optional transform to apply to features
        """
        self.database = database
        self.split = split
        self.url = url
        self.namespace = namespace
        self.username = username
        self.password = password
        self.use_cached_embeddings = use_cached_embeddings
        self.include_content = include_content
        self.transform = transform

        # Initialize feature extractor
        self.extractor = CombinedFeatureExtractor(
            user_email=user_email,
            user_context=user_context,
            include_content=include_content and not use_cached_embeddings,
        )

        # Load data synchronously (run async in sync context)
        self._emails = []
        self._load_data()

    def _load_data(self):
        """Load email data from SurrealDB."""
        async def fetch():
            db = AsyncSurreal(self.url)
            await db.signin({'username': self.username, 'password': self.password})
            await db.use(self.namespace, self.database)

            # Build query based on split
            if self.split:
                query = '''
                    SELECT
                        message_id,
                        subject,
                        body,
                        from_email,
                        to_emails,
                        cc_emails,
                        date,
                        x_from,
                        x_to,
                        action,
                        timing,
                        embedding,
                        features
                    FROM emails
                    WHERE split = $split AND action IS NOT NONE
                '''
                result = await db.query(query, {'split': self.split})
            else:
                query = '''
                    SELECT
                        message_id,
                        subject,
                        body,
                        from_email,
                        to_emails,
                        cc_emails,
                        date,
                        x_from,
                        x_to,
                        action,
                        timing,
                        embedding,
                        features
                    FROM emails
                    WHERE action IS NOT NONE
                '''
                result = await db.query(query)

            await db.close()

            # New API returns list directly
            if result and isinstance(result, list):
                return result
            return []

        # Run async function synchronously
        loop = asyncio.new_event_loop()
        try:
            self._emails = loop.run_until_complete(fetch())
        finally:
            loop.close()

    def __len__(self) -> int:
        """Return number of emails in dataset."""
        return len(self._emails)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get features and label for a single email.

        Args:
            idx: Index of email to retrieve

        Returns:
            Tuple of (features, label) where:
                - features: Float tensor
                - label: Long tensor scalar (action index)
        """
        email = self._emails[idx]

        # Check for cached features
        cached_features = email.get('features')
        cached_embedding = email.get('embedding')

        if self.use_cached_embeddings and cached_features:
            # Use cached feature vector
            features = torch.tensor(cached_features, dtype=torch.float32)
        else:
            # Build email dict for feature extraction
            email_dict = {
                'subject': email.get('subject', ''),
                'body': email.get('body', ''),
                'from': email.get('from_email', ''),
                'to': ','.join(email.get('to_emails', [])),
                'cc': ','.join(email.get('cc_emails', [])),
                'x_from': email.get('x_from', ''),
                'x_to': email.get('x_to', ''),
                'date': email.get('date', ''),
            }

            features_vec = self.extractor.to_vector(email_dict)
            features = torch.tensor(features_vec, dtype=torch.float32)

            # If we have cached embeddings and want content, append them
            if self.include_content and self.use_cached_embeddings and cached_embedding:
                embedding = torch.tensor(cached_embedding, dtype=torch.float32)
                features = torch.cat([features, embedding])

        # Get label
        action = email.get('action', 'KEPT')
        label = torch.tensor(
            ACTION_TO_IDX.get(action, ACTION_TO_IDX.get('KEPT', 4)),
            dtype=torch.long
        )

        if self.transform is not None:
            features = self.transform(features)

        return features, label

    @property
    def feature_dim(self) -> int:
        """Return dimensionality of feature vectors."""
        dim = FEATURE_DIMS['total_base']
        if self.include_content:
            dim += FEATURE_DIMS['content']
        return dim

    @property
    def num_classes(self) -> int:
        """Return number of action classes."""
        return NUM_ACTIONS

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced data."""
        counts = torch.zeros(NUM_ACTIONS, dtype=torch.float32)

        for email in self._emails:
            action = email.get('action', 'KEPT')
            label = ACTION_TO_IDX.get(action, 4)
            counts[label] += 1

        # Inverse frequency weighting with smoothing
        weights = len(self._emails) / (NUM_ACTIONS * counts + 1e-6)
        return weights

    def get_label_distribution(self) -> dict[str, int]:
        """Get distribution of action labels."""
        counts = {action: 0 for action in ACTION_TO_IDX}

        for email in self._emails:
            action = email.get('action', 'KEPT')
            if action in counts:
                counts[action] += 1

        return counts

    async def cache_embeddings(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        batch_size: int = 64,
    ):
        """Compute and cache embeddings in the database.

        This is a one-time operation that stores embeddings in SurrealDB
        for faster subsequent loading.
        """
        from src.features.content import ContentFeatureExtractor

        extractor = ContentFeatureExtractor(model_name=model_name)

        db = AsyncSurreal(self.url)
        await db.signin({'username': self.username, 'password': self.password})
        await db.use(self.namespace, self.database)

        # Process in batches
        for i in range(0, len(self._emails), batch_size):
            batch = self._emails[i:i + batch_size]

            # Build email dicts for embedding
            email_dicts = [
                {
                    'subject': e.get('subject', ''),
                    'body': e.get('body', ''),
                }
                for e in batch
            ]

            # Extract embeddings
            content_features = extractor.extract_batch(email_dicts)

            # Update database
            for email, features in zip(batch, content_features):
                message_id = email.get('message_id')
                embedding = features.embedding

                await db.query(
                    '''
                    UPDATE emails SET
                        embedding = $embedding,
                        embedding_model = $model
                    WHERE message_id = $message_id
                    ''',
                    {
                        'message_id': message_id,
                        'embedding': list(embedding),
                        'model': model_name,
                    }
                )

            print(f"Cached embeddings for {min(i + batch_size, len(self._emails))}/{len(self._emails)} emails")

        await db.close()


def create_surreal_dataloaders(
    database: str = 'enron',
    batch_size: int = 32,
    *,
    url: str = 'ws://localhost:8000/rpc',
    namespace: str = 'rl_emails',
    username: str = 'root',
    password: str = 'root',
    user_email: str = '',
    user_context: Optional[dict] = None,
    use_cached_embeddings: bool = True,
    include_content: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders from SurrealDB.

    Args:
        database: Database name ('enron' or 'gmail')
        batch_size: Batch size for training
        url: SurrealDB connection URL
        namespace: SurrealDB namespace
        username: SurrealDB username
        password: SurrealDB password
        user_email: Optional user email for feature extraction
        user_context: Optional historical context
        use_cached_embeddings: Whether to use cached embeddings from DB
        include_content: Whether to include content embeddings
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU training

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    common_args = {
        'database': database,
        'url': url,
        'namespace': namespace,
        'username': username,
        'password': password,
        'user_email': user_email,
        'user_context': user_context,
        'use_cached_embeddings': use_cached_embeddings,
        'include_content': include_content,
    }

    # Create datasets
    train_dataset = SurrealEmailDataset(split='train', **common_args)
    val_dataset = SurrealEmailDataset(split='val', **common_args)
    test_dataset = SurrealEmailDataset(split='test', **common_args)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


class SurrealGraphQueries:
    """Helper class for graph-based queries on SurrealDB.

    Provides methods for querying communication patterns and relationships.
    """

    def __init__(
        self,
        database: str = 'enron',
        url: str = 'ws://localhost:8000/rpc',
        namespace: str = 'rl_emails',
        username: str = 'root',
        password: str = 'root',
    ):
        self.database = database
        self.url = url
        self.namespace = namespace
        self.username = username
        self.password = password

    async def _connect(self) -> AsyncSurreal:
        db = AsyncSurreal(self.url)
        await db.signin({'username': self.username, 'password': self.password})
        await db.use(self.namespace, self.database)
        return db

    async def get_user_context(self, user_email: str) -> dict:
        """Get user context for feature extraction."""
        db = await self._connect()

        try:
            # Get user stats
            user_result = await db.query(
                'SELECT * FROM users WHERE email = $email LIMIT 1',
                {'email': user_email}
            )

            if not user_result or not isinstance(user_result, list) or len(user_result) == 0:
                return {}

            user = user_result[0]

            # Get communication stats
            comm_result = await db.query(
                '''
                SELECT
                    count() as total_communications,
                    math::mean(email_count) as avg_emails_per_contact
                FROM communicates
                WHERE in.email = $email
                GROUP ALL
                ''',
                {'email': user_email}
            )

            context = {
                'reply_rate_to_sender': user.get('reply_rate', 0.5),
                'avg_response_time_hours': user.get('avg_response_time_hours', 24),
                'emails_sent': user.get('emails_sent', 0),
                'emails_received': user.get('emails_received', 0),
            }

            if comm_result and isinstance(comm_result, list) and len(comm_result) > 0:
                stats = comm_result[0]
                context['total_contacts'] = stats.get('total_communications', 0)
                context['avg_emails_per_contact'] = stats.get('avg_emails_per_contact', 0)

            return context

        finally:
            await db.close()

    async def get_sender_relationship(self, user_email: str, sender_email: str) -> dict:
        """Get relationship context between user and sender."""
        db = await self._connect()

        try:
            result = await db.query(
                '''
                SELECT * FROM communicates
                WHERE in.email = $sender AND out.email = $user
                LIMIT 1
                ''',
                {'sender': sender_email, 'user': user_email}
            )

            if result and isinstance(result, list) and len(result) > 0:
                edge = result[0]
                return {
                    'emails_from_sender_30d': edge.get('email_count', 0),
                    'replies_to_sender': edge.get('reply_count', 0),
                    'avg_response_time_hours': edge.get('avg_response_time_hours'),
                }

            return {'emails_from_sender_30d': 0, 'replies_to_sender': 0}

        finally:
            await db.close()

    async def get_thread_context(self, thread_id: str) -> dict:
        """Get thread context for temporal features."""
        db = await self._connect()

        try:
            result = await db.query(
                '''
                SELECT
                    count(->belongs_to<-emails) as email_count,
                    started_at,
                    last_activity
                FROM threads
                WHERE thread_id = $thread_id
                LIMIT 1
                ''',
                {'thread_id': thread_id}
            )

            if result and isinstance(result, list) and len(result) > 0:
                thread = result[0]
                return {
                    'thread_length': thread.get('email_count', 1),
                    'thread_started': thread.get('started_at'),
                    'last_activity': thread.get('last_activity'),
                }

            return {}

        finally:
            await db.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test SurrealEmailDataset')
    parser.add_argument('--database', default='enron', help='Database name')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    print(f"Loading datasets from SurrealDB (database={args.database})...")

    try:
        train_loader, val_loader, test_loader = create_surreal_dataloaders(
            database=args.database,
            batch_size=args.batch_size,
        )

        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_loader.dataset)} emails")
        print(f"  Val:   {len(val_loader.dataset)} emails")
        print(f"  Test:  {len(test_loader.dataset)} emails")

        print(f"\nBatch sizes (batch_size={args.batch_size}):")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        print(f"  Test batches:  {len(test_loader)}")

        print(f"\nFeature dimensions: {train_loader.dataset.feature_dim}")
        print(f"Number of classes: {train_loader.dataset.num_classes}")

        print("\nLabel distribution (train):")
        dist = train_loader.dataset.get_label_distribution()
        for action, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = 100 * count / max(1, len(train_loader.dataset))
            print(f"  {action:12s}: {count:6d} ({pct:5.1f}%)")

        print("\nClass weights (train):")
        weights = train_loader.dataset.get_class_weights()
        for i, (action, _) in enumerate(sorted(ACTION_TO_IDX.items(), key=lambda x: x[1])):
            print(f"  {action:12s}: {weights[i]:.4f}")

        print("\nSample batch:")
        for features, labels in train_loader:
            print(f"  Features shape: {features.shape}")
            print(f"  Labels shape:   {labels.shape}")
            print(f"  Feature sample: {features[0, :5].tolist()}")
            print(f"  Labels sample:  {[IDX_TO_ACTION[l.item()] for l in labels[:5]]}")
            break

        print("\n✓ SurrealDB dataset test complete")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure SurrealDB is running:")
        print("  surreal start file:data/enron.db --user root --pass root")
        print("\nAnd data has been imported:")
        print("  python -m db.import_data enron data/train.json data/val.json data/test.json")
