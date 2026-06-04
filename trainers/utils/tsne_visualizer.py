import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from matplotlib.patches import Patch

# Set seaborn style for top-tier journal quality
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


class TSNEVisualizer:
    """
    t-SNE visualization tool for active learning feature distribution analysis.
    """
    
    def __init__(
        self,
        output_dir,
        n_components=2,
        perplexity=30,
        random_state=42,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
    ):
        """
        Initialize the t-SNE visualizer.
        
        Args:
            output_dir: Directory to save visualization results
            n_components: Number of dimensions for t-SNE (default: 2)
            perplexity: Perplexity parameter for t-SNE (default: 30)
            random_state: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        
        # Create output directory if it doesn't exist
        self.tsne_dir = os.path.join(output_dir, "tsne_visualization")
        os.makedirs(self.tsne_dir, exist_ok=True)
        
        # Store all rounds data for final summary
        self.all_rounds_data = []

    def _to_numpy(self, array):
        if torch.is_tensor(array):
            return array.cpu().detach().numpy()
        return np.asarray(array)

    def _get_umap(self):
        try:
            import umap  # type: ignore
        except Exception:
            return None
        return umap

    def _embed_umap(self, all_features):
        umap = self._get_umap()
        if umap is None:
            print("[UMAP] umap-learn is not installed; skipping UMAP visualization.")
            return None
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            random_state=self.random_state,
        )
        return reducer.fit_transform(all_features)

    def _save_legend_only(self, unique_classes, filename):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
        palette = sns.color_palette("husl", n_colors=len(unique_classes))
        class_colors = {cls: palette[idx] for idx, cls in enumerate(unique_classes)}
        handles = [
            Patch(facecolor=class_colors[cls], edgecolor="white", label=f"Class {cls}")
            for cls in unique_classes
        ]
        handles.append(Patch(facecolor="lightgray", edgecolor="gray", label="Unlabeled"))
        ax.axis("off")
        ax.legend(
            handles=handles,
            loc="center",
            frameon=True,
            fancybox=True,
            shadow=False,
            ncol=2,
            fontsize=10,
        )
        save_path = os.path.join(self.tsne_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close()
        print(f"Legend image saved: {save_path}")

    def _infer_feature_dim(self, *arrays):
        for arr in arrays:
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[0] > 0:
                return arr.shape[1]
            if arr.ndim == 1 and arr.size > 0:
                return arr.shape[0]
        return None

    def _as_2d(self, array, feature_dim=None):
        array = np.asarray(array)
        if array.ndim == 1:
            if array.size == 0:
                if feature_dim is None:
                    return array.reshape(0, 0)
                return array.reshape(0, feature_dim)
            return array.reshape(1, -1)
        return array
        
    def visualize_round(self, labeled_features, unlabeled_features, 
                       labeled_labels, round_idx, n_labeled):
        """
        Visualize feature distribution for a single AL round.
        
        Args:
            labeled_features: Features of labeled data (numpy array or torch tensor)
            unlabeled_features: Features of unlabeled data (numpy array or torch tensor)
            labeled_labels: True labels of labeled data (for color coding)
            round_idx: Current AL round index
            n_labeled: Number of labeled samples in this round
        """
        # Convert to numpy if needed
        labeled_features = self._to_numpy(labeled_features)
        unlabeled_features = self._to_numpy(unlabeled_features)
        labeled_labels = self._to_numpy(labeled_labels)

        feature_dim = self._infer_feature_dim(labeled_features, unlabeled_features)
        if feature_dim is None:
            print(f"[Round {round_idx}] No features available for t-SNE.")
            return

        labeled_features = self._as_2d(labeled_features, feature_dim)
        unlabeled_features = self._as_2d(unlabeled_features, feature_dim)
            
        # Combine features for t-SNE
        all_features = np.vstack([labeled_features, unlabeled_features])
        n_labeled_samples = len(labeled_features)
        n_unlabeled_samples = len(unlabeled_features)
        
        # Store for final summary
        self.all_rounds_data.append({
            'round': round_idx,
            'labeled_features': labeled_features.copy(),
            'unlabeled_features': unlabeled_features.copy(),
            'labeled_labels': labeled_labels.copy(),
            'n_labeled': n_labeled_samples
        })
        
        # Perform t-SNE with adaptive perplexity
        print(f"Performing t-SNE for Round {round_idx}...")
        # Adaptive perplexity: smaller for fewer samples, makes clusters tighter
        if len(all_features) < 2:
            print(f"[Round {round_idx}] Not enough samples for t-SNE (n={len(all_features)}).")
            return
        adaptive_perplexity = min(50, max(5, len(all_features) // 30))
        adaptive_perplexity = max(1, min(adaptive_perplexity, len(all_features) - 1))
        tsne = TSNE(n_components=self.n_components, 
                   perplexity=adaptive_perplexity,
                   random_state=self.random_state,
                   max_iter=1000,
                   learning_rate='auto',
                   init='pca')  # PCA initialization for better structure
        features_2d = tsne.fit_transform(all_features)
        print(f"  Used perplexity={adaptive_perplexity} for {len(all_features)} samples")
        
        # Split back into labeled and unlabeled
        labeled_2d = features_2d[:n_labeled_samples]
        unlabeled_2d = features_2d[n_labeled_samples:]
        
        # Create publication-quality plot
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        # Use professional color palette
        palette = sns.color_palette("husl", n_colors=len(np.unique(labeled_labels)))
        
        # Plot unlabeled data first (background) - smaller and lighter
        ax.scatter(
            unlabeled_2d[:, 0],
            unlabeled_2d[:, 1],
            c="lightgray",
            s=22,
            alpha=0.35,
            label="Unlabeled",
            edgecolors="gray",
            linewidths=0.3,
            zorder=1,
        )
        
        # Plot labeled data by class with professional colors - larger and more opaque
        for idx, class_id in enumerate(np.unique(labeled_labels)):
            mask = labeled_labels == class_id
            ax.scatter(labeled_2d[mask, 0], labeled_2d[mask, 1],
                      c=[palette[idx]], s=60, alpha=0.85,
                      label=f'Class {class_id}', 
                      edgecolors='white', linewidths=0.8, zorder=2)
        
        # Styling
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(
            f"Round {round_idx} | Labeled: {n_labeled_samples} | Unlabeled: {n_unlabeled_samples}",
            fontsize=13,
            fontweight="bold",
            pad=16,
        )
        
        # Legend with better positioning
        ax.legend(loc='best', frameon=True, fancybox=True, 
                 shadow=True, ncol=2, fontsize=10)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.tsne_dir, f"round_{round_idx}_tsne.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE visualization saved: {save_path}")

    def visualize_round_umap(
        self,
        labeled_features,
        unlabeled_features,
        labeled_labels,
        round_idx,
        n_labeled,
    ):
        """
        UMAP visualization for a single AL round (same plotting style as t-SNE).
        """
        labeled_features = self._to_numpy(labeled_features)
        unlabeled_features = self._to_numpy(unlabeled_features)
        labeled_labels = self._to_numpy(labeled_labels)

        feature_dim = self._infer_feature_dim(labeled_features, unlabeled_features)
        if feature_dim is None:
            print(f"[Round {round_idx}] No features available for UMAP.")
            return

        labeled_features = self._as_2d(labeled_features, feature_dim)
        unlabeled_features = self._as_2d(unlabeled_features, feature_dim)

        all_features = np.vstack([labeled_features, unlabeled_features])
        n_labeled_samples = len(labeled_features)
        n_unlabeled_samples = len(unlabeled_features)

        print(f"Performing UMAP for Round {round_idx}...")
        if len(all_features) < 2:
            print(f"[Round {round_idx}] Not enough samples for UMAP (n={len(all_features)}).")
            return

        features_2d = self._embed_umap(all_features)
        if features_2d is None:
            return

        labeled_2d = features_2d[:n_labeled_samples]
        unlabeled_2d = features_2d[n_labeled_samples:]

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        palette = sns.color_palette("husl", n_colors=len(np.unique(labeled_labels)))

        ax.scatter(
            unlabeled_2d[:, 0],
            unlabeled_2d[:, 1],
            c="lightgray",
            s=18,
            alpha=0.35,
            label="Unlabeled",
            edgecolors="gray",
            linewidths=0.3,
            zorder=1,
        )

        for idx, class_id in enumerate(np.unique(labeled_labels)):
            mask = labeled_labels == class_id
            ax.scatter(
                labeled_2d[mask, 0],
                labeled_2d[mask, 1],
                c=[palette[idx]],
                s=60,
                alpha=0.85,
                label=f"Class {class_id}",
                edgecolors="white",
                linewidths=0.8,
                zorder=2,
            )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(
            f"Round {round_idx} | Labeled: {n_labeled_samples} | Unlabeled: {n_unlabeled_samples}",
            fontsize=13,
            fontweight="bold",
            pad=16,
        )

        ax.legend(loc="best", frameon=True, fancybox=True, shadow=True, ncol=2, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        save_path = os.path.join(self.tsne_dir, f"round_{round_idx}_umap.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"UMAP visualization saved: {save_path}")
        
    def create_summary_visualization(self):
        """
        Create a comprehensive summary visualization showing evolution across rounds.
        Uses a single t-SNE for all rounds to maintain consistent coordinate system.
        """
        if len(self.all_rounds_data) == 0:
            print("No data available for summary visualization.")
            return
            
        n_rounds = len(self.all_rounds_data)
        
        print("Preparing data for unified t-SNE across all rounds...")
        
        # ========== STEP 1: Collect ALL features from ALL rounds ==========
        all_round_features = []
        all_round_labels = []
        all_round_types = []  # 0=unlabeled, 1=labeled
        all_round_nums = []   # which round
        
        total_labeled_counts = []
        total_unlabeled_counts = []

        rng = np.random.default_rng(self.random_state)
        for round_idx, data in enumerate(self.all_rounds_data):
            labeled_features = data['labeled_features']
            unlabeled_features = data['unlabeled_features']
            labeled_labels = data['labeled_labels']

            # Record true counts (before subsampling for visualization)
            total_labeled_counts.append(len(labeled_features))
            total_unlabeled_counts.append(len(unlabeled_features))
            
            # Add labeled samples
            all_round_features.append(labeled_features)
            all_round_labels.extend(labeled_labels)
            all_round_types.extend([1] * len(labeled_features))
            all_round_nums.extend([round_idx] * len(labeled_features))
            
            # Add unlabeled samples (limited to avoid too many points)
            max_unlabeled_per_round = 500  # Limit unlabeled samples per round
            if len(unlabeled_features) > max_unlabeled_per_round:
                indices = rng.choice(len(unlabeled_features), max_unlabeled_per_round, replace=False)
                unlabeled_features = unlabeled_features[indices]
            
            all_round_features.append(unlabeled_features)
            all_round_labels.extend([-1] * len(unlabeled_features))  # -1 for unlabeled
            all_round_types.extend([0] * len(unlabeled_features))
            all_round_nums.extend([round_idx] * len(unlabeled_features))
        
        # Combine all features
        all_features = np.vstack(all_round_features)
        all_round_labels = np.array(all_round_labels)
        all_round_types = np.array(all_round_types)
        all_round_nums = np.array(all_round_nums)
        
        print(f"Total samples for t-SNE: {len(all_features)}")
        
        # ========== STEP 2: Perform ONE unified t-SNE for all rounds ==========
        # Adjust perplexity based on data size
        if len(all_features) < 2:
            print("Not enough samples for unified t-SNE.")
            return
        perplexity = min(50, max(5, len(all_features) // 50))
        perplexity = max(1, min(perplexity, len(all_features) - 1))
        print(f"Running unified t-SNE with perplexity={perplexity}...")
        
        tsne = TSNE(n_components=2, 
                   perplexity=perplexity,
                   random_state=self.random_state,
                   max_iter=1000,
                   learning_rate='auto',
                   init='pca')
        features_2d = tsne.fit_transform(all_features)
        
        # ========== STEP 3: Create visualization for each round ==========
        fig = plt.figure(figsize=(6*n_rounds, 7), dpi=300)
        gs = fig.add_gridspec(2, n_rounds, height_ratios=[6, 1], hspace=0.3, wspace=0.25)
        
        # Use consistent color palette
        unique_classes = np.unique(all_round_labels[all_round_labels != -1])
        palette = sns.color_palette("husl", n_colors=len(unique_classes))
        class_colors = {cls: palette[idx] for idx, cls in enumerate(unique_classes)}
        
        for round_idx in range(n_rounds):
            ax = fig.add_subplot(gs[0, round_idx])
            
            # Filter data for this round
            round_mask = all_round_nums == round_idx
            round_features_2d = features_2d[round_mask]
            round_labels = all_round_labels[round_mask]
            round_types = all_round_types[round_mask]
            
            # Split into labeled and unlabeled
            unlabeled_mask = round_types == 0
            labeled_mask = round_types == 1
            
            unlabeled_2d = round_features_2d[unlabeled_mask]
            labeled_2d = round_features_2d[labeled_mask]
            labeled_labels_round = round_labels[labeled_mask]
            
            # Use true totals in titles (not the subsampled counts)
            n_labeled = total_labeled_counts[round_idx]
            n_unlabeled = total_unlabeled_counts[round_idx]
            
            # Plot unlabeled in background (smaller, more transparent)
            if len(unlabeled_2d) > 0:
                ax.scatter(
                    unlabeled_2d[:, 0],
                    unlabeled_2d[:, 1],
                    c="lightgray",
                    s=14,
                    alpha=0.35,
                    edgecolors="gray",
                    linewidths=0.25,
                    label="Unlabeled" if round_idx == 0 else "",
                    zorder=1,
                )
            
            # Plot labeled with consistent colors (larger, more opaque)
            for class_id in unique_classes:
                mask = labeled_labels_round == class_id
                if np.any(mask):
                    ax.scatter(labeled_2d[mask, 0], labeled_2d[mask, 1],
                              c=[class_colors[class_id]], s=60, alpha=0.9,
                              edgecolors='white', linewidths=0.8,
                              label=f'Class {class_id}' if round_idx == 0 else '',
                              zorder=2)
            
            # Title with round info
            ax.set_title(
                f"Round {round_idx} | Labeled: {n_labeled} | Unlabeled: {n_unlabeled}",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            # Statistics bar below each round
            ax_stat = fig.add_subplot(gs[1, round_idx])
            ax_stat.axis('off')
            
            # Calculate class distribution for this round's labeled data
            if len(labeled_labels_round) > 0:
                unique, counts = np.unique(labeled_labels_round, return_counts=True)
                stats_text = f"Classes: {len(unique)}\n"
                stats_text += f"Samples per class:\n"
                stats_text += f"Min: {counts.min()}, Max: {counts.max()}\n"
                stats_text += f"Avg: {counts.mean():.1f} ± {counts.std():.1f}"
            else:
                stats_text = "No labeled data"
            
            ax_stat.text(0.5, 0.5, stats_text,
                        ha='center', va='center',
                        fontsize=9, family='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='lightblue', alpha=0.3))
        
        # Save summary
        summary_path = os.path.join(self.tsne_dir, "summary_all_rounds.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Enhanced summary visualization saved: {summary_path}")
        print(f"   Used unified t-SNE projection with perplexity={perplexity}")

        # Save legend-only image
        self._save_legend_only(unique_classes, "legend_only.png")

    def create_summary_visualization_umap(self):
        """
        Create a comprehensive summary visualization using a single UMAP across all rounds.
        """
        if len(self.all_rounds_data) == 0:
            print("No data available for UMAP summary visualization.")
            return

        n_rounds = len(self.all_rounds_data)
        print("Preparing data for unified UMAP across all rounds...")

        all_round_features = []
        all_round_labels = []
        all_round_types = []  # 0=unlabeled, 1=labeled
        all_round_nums = []
        total_labeled_counts = []
        total_unlabeled_counts = []

        rng = np.random.default_rng(self.random_state)
        for round_idx, data in enumerate(self.all_rounds_data):
            labeled_features = data["labeled_features"]
            unlabeled_features = data["unlabeled_features"]
            labeled_labels = data["labeled_labels"]

            total_labeled_counts.append(len(labeled_features))
            total_unlabeled_counts.append(len(unlabeled_features))

            all_round_features.append(labeled_features)
            all_round_labels.extend(labeled_labels)
            all_round_types.extend([1] * len(labeled_features))
            all_round_nums.extend([round_idx] * len(labeled_features))

            max_unlabeled_per_round = 500
            if len(unlabeled_features) > max_unlabeled_per_round:
                indices = rng.choice(len(unlabeled_features), max_unlabeled_per_round, replace=False)
                unlabeled_features = unlabeled_features[indices]

            all_round_features.append(unlabeled_features)
            all_round_labels.extend([-1] * len(unlabeled_features))
            all_round_types.extend([0] * len(unlabeled_features))
            all_round_nums.extend([round_idx] * len(unlabeled_features))

        all_features = np.vstack(all_round_features)
        all_round_labels = np.array(all_round_labels)
        all_round_types = np.array(all_round_types)
        all_round_nums = np.array(all_round_nums)

        print(f"Total samples for UMAP: {len(all_features)}")
        if len(all_features) < 2:
            print("Not enough samples for unified UMAP.")
            return

        features_2d = self._embed_umap(all_features)
        if features_2d is None:
            return

        fig = plt.figure(figsize=(6 * n_rounds, 7), dpi=300)
        gs = fig.add_gridspec(2, n_rounds, height_ratios=[6, 1], hspace=0.3, wspace=0.25)

        unique_classes = np.unique(all_round_labels[all_round_labels != -1])
        palette = sns.color_palette("husl", n_colors=len(unique_classes))
        class_colors = {cls: palette[idx] for idx, cls in enumerate(unique_classes)}

        for round_idx in range(n_rounds):
            ax = fig.add_subplot(gs[0, round_idx])

            round_mask = all_round_nums == round_idx
            round_features_2d = features_2d[round_mask]
            round_labels = all_round_labels[round_mask]
            round_types = all_round_types[round_mask]

            unlabeled_mask = round_types == 0
            labeled_mask = round_types == 1

            unlabeled_2d = round_features_2d[unlabeled_mask]
            labeled_2d = round_features_2d[labeled_mask]
            labeled_labels_round = round_labels[labeled_mask]

            n_labeled = total_labeled_counts[round_idx]
            n_unlabeled = total_unlabeled_counts[round_idx]

            if len(unlabeled_2d) > 0:
                ax.scatter(
                    unlabeled_2d[:, 0],
                    unlabeled_2d[:, 1],
                    c="lightgray",
                    s=14,
                    alpha=0.35,
                    edgecolors="gray",
                    linewidths=0.25,
                    label="Unlabeled" if round_idx == 0 else "",
                    zorder=1,
                )

            for class_id in unique_classes:
                mask = labeled_labels_round == class_id
                if np.any(mask):
                    ax.scatter(
                        labeled_2d[mask, 0],
                        labeled_2d[mask, 1],
                        c=[class_colors[class_id]],
                        s=60,
                        alpha=0.9,
                        edgecolors="white",
                        linewidths=0.8,
                        label=f"Class {class_id}" if round_idx == 0 else "",
                        zorder=2,
                    )

            ax.set_title(
                f"Round {round_idx} | Labeled: {n_labeled} | Unlabeled: {n_unlabeled}",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.2, linestyle="--")

            ax_stat = fig.add_subplot(gs[1, round_idx])
            ax_stat.axis("off")
            if len(labeled_labels_round) > 0:
                unique, counts = np.unique(labeled_labels_round, return_counts=True)
                stats_text = f"Classes: {len(unique)}\n"
                stats_text += "Samples per class:\n"
                stats_text += f"Min: {counts.min()}, Max: {counts.max()}\n"
                stats_text += f"Avg: {counts.mean():.1f} ± {counts.std():.1f}"
            else:
                stats_text = "No labeled data"

            ax_stat.text(
                0.5,
                0.5,
                stats_text,
                ha="center",
                va="center",
                fontsize=9,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3),
            )

        summary_path = os.path.join(self.tsne_dir, "summary_all_rounds_umap.png")
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ UMAP summary visualization saved: {summary_path}")

        # Save legend-only image for UMAP summary (same classes)
        self._save_legend_only(unique_classes, "legend_only_umap.png")
        
    def visualize_with_query_samples(self, labeled_features, unlabeled_features,
                                    query_features, labeled_labels, query_labels,
                                    round_idx):
        """
        Visualize with highlighted query samples (newly selected samples).
        
        Args:
            labeled_features: Features of existing labeled data
            unlabeled_features: Features of remaining unlabeled data
            query_features: Features of newly queried samples
            labeled_labels: Labels of existing labeled data
            query_labels: Predicted labels of queried samples
            round_idx: Current AL round index
        """
        # Convert to numpy if needed
        labeled_features = self._to_numpy(labeled_features)
        unlabeled_features = self._to_numpy(unlabeled_features)
        query_features = self._to_numpy(query_features)
        labeled_labels = self._to_numpy(labeled_labels)
        query_labels = self._to_numpy(query_labels)

        feature_dim = self._infer_feature_dim(labeled_features, unlabeled_features, query_features)
        if feature_dim is None:
            print(f"[Round {round_idx}] No features available for t-SNE (query view).")
            return

        labeled_features = self._as_2d(labeled_features, feature_dim)
        unlabeled_features = self._as_2d(unlabeled_features, feature_dim)
        query_features = self._as_2d(query_features, feature_dim)
            
        # Combine all features
        all_features = np.vstack([labeled_features, unlabeled_features, query_features])
        n_labeled = len(labeled_features)
        n_unlabeled = len(unlabeled_features)
        n_query = len(query_features)
        
        # Perform t-SNE
        print(f"Performing t-SNE with query samples for Round {round_idx}...")
        if len(all_features) < 2:
            print(f"[Round {round_idx}] Not enough samples for t-SNE (n={len(all_features)}).")
            return
        tsne = TSNE(n_components=self.n_components,
                   perplexity=max(1, min(self.perplexity, len(all_features) - 1)),
                   random_state=self.random_state,
                   max_iter=1000)
        features_2d = tsne.fit_transform(all_features)
        
        # Split results
        labeled_2d = features_2d[:n_labeled]
        unlabeled_2d = features_2d[n_labeled:n_labeled+n_unlabeled]
        query_2d = features_2d[n_labeled+n_unlabeled:]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        # Plot unlabeled (background)
        ax.scatter(unlabeled_2d[:, 0], unlabeled_2d[:, 1],
                  c='lightgray', s=30, alpha=0.2,
                  label='Unlabeled', edgecolors='none')
        
        # Plot labeled
        palette = sns.color_palette("husl", n_colors=len(np.unique(labeled_labels)))
        for idx, class_id in enumerate(np.unique(labeled_labels)):
            mask = labeled_labels == class_id
            ax.scatter(labeled_2d[mask, 0], labeled_2d[mask, 1],
                      c=[palette[idx]], s=50, alpha=0.6,
                      label=f'Labeled Class {class_id}',
                      edgecolors='white', linewidths=0.5)
        
        # Plot queried samples with star markers
        ax.scatter(query_2d[:, 0], query_2d[:, 1],
                  c='red', s=200, alpha=0.8, marker='*',
                  label='Queried Samples',
                  edgecolors='darkred', linewidths=1.5)
        
        # Styling
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
        ax.set_title(f'Round {round_idx}: Query Sample Selection\n'
                    f'Labeled: {n_labeled} | Unlabeled: {n_unlabeled} | Queried: {n_query}',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(loc='best', frameon=True, fancybox=True,
                 shadow=True, ncol=2, fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.tsne_dir, f"round_{round_idx}_query_highlight.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Query highlight visualization saved: {save_path}")
