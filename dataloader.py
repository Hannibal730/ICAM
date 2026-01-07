import os
import logging
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, datasets
import re # re 모듈 임포트

# =============================================================================
# 1. 데이터셋 클래스 정의
# =============================================================================
class CustomImageDataset(Dataset):
    """CSV 파일과 이미지 폴더 경로를 받아 데이터를 로드하는 커스텀 데이터셋입니다."""
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # 클래스 이름과 인덱스를 매핑합니다. (ImageFolder와 호환)
        self.classes = ['Normal', 'Defect']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        # 'Defect' 열의 값을 명시적으로 레이블로 사용합니다.
        label = int(self.img_labels.loc[idx, 'Defect'])
        return image, label, img_name

class InferenceImageDataset(Dataset):
    """정답 레이블 없이, 지정된 폴더의 모든 이미지를 로드하는 추론 전용 데이터셋입니다."""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # 지원하는 이미지 확장자
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        self.img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in self.image_extensions]
        if not self.img_files:
            logging.warning(f"'{img_dir}'에서 이미지를 찾을 수 없습니다.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image, -1, img_name # 레이블은 -1과 같은 placeholder 값으로 반환

class ImageFolderWithPaths(datasets.ImageFolder):
    """기존 ImageFolder에 파일 경로(파일명)를 함께 반환하는 기능을 추가한 클래스입니다."""
    def __getitem__(self, index):
        # 기존 ImageFolder의 __getitem__을 호출하여 이미지와 레이블을 가져옵니다.
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # 파일 경로를 가져옵니다.
        path = self.imgs[index][0]
        
        # [수정] 파일명에서 불필요한 문자열 제거
        # 파일명에서 유니코드 이스케이프를 디코딩하고, 이전의 ' '로 시작하는 불필요한 접두사 제거 로직은 제거합니다.
        # PC의 파일명과 일관성을 유지하기 위해 접두사를 제거하지 않습니다.
        filename = _decode_unicode_escapes_in_filename(os.path.basename(path))
        return (*original_tuple, filename)

def _decode_unicode_escapes_in_filename(s):
    """
    #Uxxxx 형태의 유니코드 이스케이프 시퀀스를 실제 유니코드 문자로 디코딩합니다.
    파일 이름 중간이나 여러 번 나타나는 경우를 처리합니다.
    """
    def replace_hex_escape(match):
        hex_code = match.group(1)
        try:
            return chr(int(hex_code, 16))
        except ValueError:
            return match.group(0) # 유효하지 않은 16진수 코드인 경우 원본 반환
    
    return re.sub(r'#U([0-9a-fA-F]{4})', replace_hex_escape, s)

class TransformedSubset(Dataset):
    """특정 Subset에 transform을 적용하기 위한 래퍼 클래스입니다."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        # Subset의 classes와 class_to_idx 속성을 상속받아 DataLoader와의 호환성을 유지합니다.
        if hasattr(subset.dataset, 'classes'):
            self.classes = subset.dataset.classes
        if hasattr(subset.dataset, 'class_to_idx'):
            self.class_to_idx = subset.dataset.class_to_idx

    def __getitem__(self, index):
        # Subset에서 (PIL Image, label, filename)을 가져옵니다.
        image, label, filename = self.subset[index]
        
        # transform을 적용합니다.
        if self.transform:
            image = self.transform(image)
        
        return image, label, filename

    def __len__(self):
        return len(self.subset)

# =============================================================================
# 2. 데이터 준비 함수
# =============================================================================
def prepare_data(run_cfg, train_cfg, model_cfg):
    """데이터셋을 로드하고 전처리하여 DataLoader를 생성합니다."""
    img_size = model_cfg.img_size
    dataset_cfg = run_cfg.dataset

    # --- 데이터 샘플링 로직 함수 ---
    def get_subset(dataset, name, sampling_ratios, random_seed):
        """데이터셋에서 지정된 비율만큼 단순 랜덤 샘플링을 수행합니다."""
        ratio = 1.0
        if isinstance(sampling_ratios, dict):
            ratio = sampling_ratios.get(name, 1.0)
        elif isinstance(sampling_ratios, (float, int)):
            ratio = sampling_ratios

        if ratio < 1.0:
            logging.info(f"'{name}' 데이터셋을 {ratio * 100:.1f}% 비율로 샘플링합니다 (random_seed={random_seed}).")
            num_total = len(dataset)
            num_to_sample = int(num_total * ratio)
            num_to_sample = max(1, num_to_sample)
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(num_total, size=num_to_sample, replace=False)
            return Subset(dataset, indices)
        return dataset

    # --- 데이터 변환(Transform) 정의 ---
    # Sewer-ML의 lightning_trainer.py에서 사용하는 전처리 가져오기
    normalize = transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    valid_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    try:
        logging.info(f"'{dataset_cfg.name}' 데이터 로드를 시작합니다 (Type: {dataset_cfg.type}).")
        
        # --- 데이터셋 로드 ---
        if dataset_cfg.type == 'csv':
            full_train_dataset = CustomImageDataset(csv_file=dataset_cfg.paths['train_csv'], img_dir=dataset_cfg.paths['train_img_dir'], transform=train_transform)
            full_valid_dataset = CustomImageDataset(csv_file=dataset_cfg.paths['valid_csv'], img_dir=dataset_cfg.paths['valid_img_dir'], transform=valid_test_transform)
            full_test_dataset = CustomImageDataset(csv_file=dataset_cfg.paths['test_csv'], img_dir=dataset_cfg.paths['test_img_dir'], transform=valid_test_transform)
            class_names = full_train_dataset.classes
        elif dataset_cfg.type == 'image_folder':
            logging.info(f"'{dataset_cfg.paths['img_folder']}' 경로의 데이터를 훈련/테스트셋으로 분할합니다.")
            
            # [수정] 데이터 분할의 일관성을 보장하기 위해, transform이 없는 단일 데이터셋을 먼저 생성합니다.
            base_dataset = ImageFolderWithPaths(root=dataset_cfg.paths['img_folder'], transform=None) # transform=None
            
            # [최종 수정] 파일 경로를 직접 수정하지 않고, 정렬 시에만 정규화된 파일명을 사용합니다.
            # 1. 파일명에 포함된 유니코드 이스케이프 시퀀스(#Uxxxx)를 실제 문자로 디코딩합니다.
            # 2. 전체 경로가 아닌 파일명(basename)을 기준으로 정렬합니다.
            # 이 두 가지를 통해 서로 다른 OS(Windows, Linux)에서도 동일한 파일 순서를 보장하여
            # random_split이 항상 동일한 테스트셋을 생성하도록 합니다.
            # 원본 파일 경로는 그대로 유지하여 파일 로딩 오류를 방지합니다.
            base_dataset.imgs.sort(key=lambda item: _decode_unicode_escapes_in_filename(os.path.basename(item[0])))

            num_total = len(base_dataset)
            train_ratio = getattr(dataset_cfg, 'train_split_ratio', 0.8)
            num_train = int(num_total * train_ratio)
            num_test = num_total - num_train

            # 데이터 분할 시 global_seed를 사용합니다.
            global_seed = getattr(run_cfg, 'global_seed', 42) # global_seed가 없으면 기본값 42 사용
            logging.info(f"총 {num_total}개 데이터를 훈련용 {num_train}개, 테스트용 {num_test}개로 분할합니다 (split_seed={global_seed}).")

            # [수정] 단일 데이터셋을 기준으로 분할합니다.
            generator = torch.Generator().manual_seed(global_seed)
            train_subset, test_subset = random_split(base_dataset, [num_train, num_test], generator=generator)

            # [수정] 분할된 Subset에 각각 다른 transform을 적용하는 래퍼를 사용합니다.
            full_train_dataset = TransformedSubset(train_subset, transform=train_transform)
            full_test_dataset = TransformedSubset(test_subset, transform=valid_test_transform)
            full_valid_dataset = full_test_dataset # 검증셋은 테스트셋과 동일하게 사용
            
            class_names = base_dataset.classes
        else:
            raise ValueError(f"지원하지 않는 데이터셋 타입입니다: {dataset_cfg.type}")

        num_labels = len(class_names)
        logging.info(f"데이터셋 클래스: {class_names} (총 {num_labels}개)")

        # --- 데이터 샘플링 ---
        sampling_ratios = getattr(run_cfg, 'random_sampling_ratio', None)
        global_seed = getattr(run_cfg, 'global_seed', 42) # global_seed가 없으면 기본값 42 사용
        train_dataset = get_subset(full_train_dataset, 'train', sampling_ratios, global_seed)
        valid_dataset = get_subset(full_valid_dataset, 'valid', sampling_ratios, global_seed)
        test_dataset = get_subset(full_test_dataset, 'test', sampling_ratios, global_seed)

        # --- DataLoader 생성 ---
        # ImageFolder는 (image, label)을 반환하므로, CustomImageDataset과 형식을 맞추기 위해 collate_fn을 사용합니다.
        def collate_fn(batch):
            # 모든 데이터셋 클래스가 (image, label, filename) 튜플을 반환하도록 통일되었습니다.
            images, labels, filenames = zip(*batch)
            return torch.stack(images, 0), torch.tensor(labels), list(filenames)

        train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False, collate_fn=collate_fn, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False, collate_fn=collate_fn)
        
        # --- BCE 손실 함수를 위한 pos_weight 계산 ---
        pos_weight = None # 기본값은 None
        loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()

        if loss_function_name == 'bcewithlogitsloss' and getattr(train_cfg, 'bce_pos_weight', None) == 'auto':
            logging.info("BCE 손실 함수의 'pos_weight'를 자동 계산합니다.")
            labels = []
            
            # [수정] full_train_dataset이 Subset 또는 TransformedSubset 객체인지 확인합니다.
            subset_to_process = None
            if hasattr(full_train_dataset, 'subset') and isinstance(getattr(full_train_dataset, 'subset'), Subset):
                # TransformedSubset의 경우
                subset_to_process = full_train_dataset.subset
            elif isinstance(full_train_dataset, Subset):
                # 일반 Subset의 경우
                subset_to_process = full_train_dataset

            if subset_to_process:
                original_dataset = subset_to_process.dataset
                indices = subset_to_process.indices
            else: # Subset이 아니라면 (예: CustomImageDataset) 자기 자신을 원본으로 사용합니다.
                original_dataset = full_train_dataset
                indices = list(range(len(original_dataset)))

            if dataset_cfg.type == 'csv':
                # CustomImageDataset의 경우, 'Defect' 열이 레이블입니다.
                labels = original_dataset.img_labels.iloc[indices]['Defect'].values
            elif dataset_cfg.type == 'image_folder':
                # ImageFolder의 경우, 클래스 이름을 기반으로 레이블을 결정합니다.
                # 데이터셋 이름에 따라 'Defect' 또는 'abnormal'을 양성 클래스로 간주합니다.
                if dataset_cfg.name in ['Sewer-TAP', 'Sewer-TAPNEW']:
                    positive_class_name = 'abnormal'
                else: # Sewer-ML 및 기타
                    positive_class_name = 'Defect'
                
                try:
                    positive_class_idx = original_dataset.class_to_idx[positive_class_name]
                    all_labels = np.array(original_dataset.targets)
                    subset_labels = all_labels[indices]
                    labels = (subset_labels == positive_class_idx).astype(int)
                except KeyError:
                    logging.warning(f"'{positive_class_name}' 클래스를 찾을 수 없어 pos_weight 계산을 건너뜁니다. 클래스: {original_dataset.classes}")
                    labels = []
            
            if len(labels) > 0:
                pos_count = np.sum(labels)
                neg_count = len(labels) - pos_count
                if pos_count > 0:
                    pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float)
                    logging.info(f"계산된 pos_weight: {pos_weight.item():.4f} (Negative: {neg_count}, Positive: {pos_count})")

        logging.info(f"훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(valid_dataset)}개, 테스트 데이터: {len(test_dataset)}개")
        
        return train_loader, valid_loader, test_loader, num_labels, class_names, pos_weight # pos_weight 반환 추가
        
    except FileNotFoundError as e:
        logging.error(f"데이터 폴더 또는 CSV 파일을 찾을 수 없습니다: {e}. 'config.yaml'의 경로 설정을 확인해주세요.")
        exit()
    except Exception as e:
        logging.error(f"데이터 준비 중 오류 발생: {e}")
        exit()