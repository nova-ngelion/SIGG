// src/file_operations.rs
// 完全なファイル操作システム

use crate::error::SiggError;
use std::fs;
use std::path::{Path, PathBuf};
use std::io::{Read, Write};

pub struct FileOperations {
    base_path: PathBuf,
    backup_enabled: bool,
}

impl FileOperations {
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            backup_enabled: true,
        }
    }

    /// ファイル読み込み
    pub fn read_file(&self, path: impl AsRef<Path>) -> Result<String, SiggError> {
        let full_path = self.resolve_path(path)?;
        
        fs::read_to_string(&full_path)
            .map_err(|e| SiggError::runtime(format!("ファイル読み込みエラー: {}", e)))
    }

    /// バイナリファイル読み込み
    pub fn read_binary(&self, path: impl AsRef<Path>) -> Result<Vec<u8>, SiggError> {
        let full_path = self.resolve_path(path)?;
        
        fs::read(&full_path)
            .map_err(|e| SiggError::runtime(format!("バイナリ読み込みエラー: {}", e)))
    }

    /// ファイル書き込み
    pub fn write_file(
        &self,
        path: impl AsRef<Path>,
        content: &str,
    ) -> Result<(), SiggError> {
        let full_path = self.resolve_path(path)?;
        
        // バックアップ作成
        if self.backup_enabled && full_path.exists() {
            self.create_backup(&full_path)?;
        }
        
        // 親ディレクトリを作成
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| SiggError::runtime(format!("ディレクトリ作成エラー: {}", e)))?;
        }
        
        fs::write(&full_path, content)
            .map_err(|e| SiggError::runtime(format!("ファイル書き込みエラー: {}", e)))
    }

    /// バイナリファイル書き込み
    pub fn write_binary(
        &self,
        path: impl AsRef<Path>,
        data: &[u8],
    ) -> Result<(), SiggError> {
        let full_path = self.resolve_path(path)?;
        
        if self.backup_enabled && full_path.exists() {
            self.create_backup(&full_path)?;
        }
        
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| SiggError::runtime(format!("ディレクトリ作成エラー: {}", e)))?;
        }
        
        fs::write(&full_path, data)
            .map_err(|e| SiggError::runtime(format!("バイナリ書き込みエラー: {}", e)))
    }

    /// ファイル削除
    pub fn delete_file(&self, path: impl AsRef<Path>) -> Result<(), SiggError> {
        let full_path = self.resolve_path(path)?;
        
        if self.backup_enabled && full_path.exists() {
            self.create_backup(&full_path)?;
        }
        
        fs::remove_file(&full_path)
            .map_err(|e| SiggError::runtime(format!("ファイル削除エラー: {}", e)))
    }

    /// ファイル移動
    pub fn move_file(
        &self,
        from: impl AsRef<Path>,
        to: impl AsRef<Path>,
    ) -> Result<(), SiggError> {
        let from_path = self.resolve_path(from)?;
        let to_path = self.resolve_path(to)?;
        
        if self.backup_enabled && to_path.exists() {
            self.create_backup(&to_path)?;
        }
        
        if let Some(parent) = to_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| SiggError::runtime(format!("ディレクトリ作成エラー: {}", e)))?;
        }
        
        fs::rename(&from_path, &to_path)
            .map_err(|e| SiggError::runtime(format!("ファイル移動エラー: {}", e)))
    }

    /// ファイルコピー
    pub fn copy_file(
        &self,
        from: impl AsRef<Path>,
        to: impl AsRef<Path>,
    ) -> Result<(), SiggError> {
        let from_path = self.resolve_path(from)?;
        let to_path = self.resolve_path(to)?;
        
        if self.backup_enabled && to_path.exists() {
            self.create_backup(&to_path)?;
        }
        
        if let Some(parent) = to_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| SiggError::runtime(format!("ディレクトリ作成エラー: {}", e)))?;
        }
        
        fs::copy(&from_path, &to_path)
            .map_err(|e| SiggError::runtime(format!("ファイルコピーエラー: {}", e)))?;
        
        Ok(())
    }

    /// ディレクトリ作成
    pub fn create_directory(&self, path: impl AsRef<Path>) -> Result<(), SiggError> {
        let full_path = self.resolve_path(path)?;
        
        fs::create_dir_all(&full_path)
            .map_err(|e| SiggError::runtime(format!("ディレクトリ作成エラー: {}", e)))
    }

    /// ディレクトリ削除
    pub fn delete_directory(&self, path: impl AsRef<Path>) -> Result<(), SiggError> {
        let full_path = self.resolve_path(path)?;
        
        fs::remove_dir_all(&full_path)
            .map_err(|e| SiggError::runtime(format!("ディレクトリ削除エラー: {}", e)))
    }

    /// ディレクトリ内容リスト
    pub fn list_directory(&self, path: impl AsRef<Path>) -> Result<Vec<PathBuf>, SiggError> {
        let full_path = self.resolve_path(path)?;
        
        let entries = fs::read_dir(&full_path)
            .map_err(|e| SiggError::runtime(format!("ディレクトリ読み込みエラー: {}", e)))?;
        
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry
                .map_err(|e| SiggError::runtime(format!("エントリ読み込みエラー: {}", e)))?;
            paths.push(entry.path());
        }
        
        Ok(paths)
    }

    /// ファイル存在確認
    pub fn exists(&self, path: impl AsRef<Path>) -> Result<bool, SiggError> {
        let full_path = self.resolve_path(path)?;
        Ok(full_path.exists())
    }

    /// ファイル情報取得
    pub fn file_info(&self, path: impl AsRef<Path>) -> Result<FileInfo, SiggError> {
        let full_path = self.resolve_path(path)?;
        
        let metadata = fs::metadata(&full_path)
            .map_err(|e| SiggError::runtime(format!("メタデータ取得エラー: {}", e)))?;
        
        Ok(FileInfo {
            path: full_path,
            size: metadata.len(),
            is_file: metadata.is_file(),
            is_dir: metadata.is_dir(),
            modified: metadata.modified().ok(),
        })
    }

    /// パターンマッチングでファイル検索
    pub fn find_files(
        &self,
        pattern: &str,
        search_path: impl AsRef<Path>,
    ) -> Result<Vec<PathBuf>, SiggError> {
        let full_path = self.resolve_path(search_path)?;
        let mut results = Vec::new();
        
        self.find_files_recursive(&full_path, pattern, &mut results)?;
        
        Ok(results)
    }

    fn find_files_recursive(
        &self,
        dir: &Path,
        pattern: &str,
        results: &mut Vec<PathBuf>,
    ) -> Result<(), SiggError> {
        if !dir.is_dir() {
            return Ok(());
        }
        
        let entries = fs::read_dir(dir)
            .map_err(|e| SiggError::runtime(format!("ディレクトリ読み込みエラー: {}", e)))?;
        
        for entry in entries {
            let entry = entry
                .map_err(|e| SiggError::runtime(format!("エントリ読み込みエラー: {}", e)))?;
            let path = entry.path();
            
            if path.is_dir() {
                self.find_files_recursive(&path, pattern, results)?;
            } else if let Some(name) = path.file_name() {
                if name.to_string_lossy().contains(pattern) {
                    results.push(path);
                }
            }
        }
        
        Ok(())
    }

    /// バックアップ作成
    fn create_backup(&self, path: &Path) -> Result<(), SiggError> {
        if !path.exists() {
            return Ok(());
        }
        
        let backup_path = path.with_extension(format!(
            "{}.backup",
            path.extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
        ));
        
        fs::copy(path, &backup_path)
            .map_err(|e| SiggError::runtime(format!("バックアップ作成エラー: {}", e)))?;
        
        Ok(())
    }

    /// パス解決
    fn resolve_path(&self, path: impl AsRef<Path>) -> Result<PathBuf, SiggError> {
        let path = path.as_ref();
        
        if path.is_absolute() {
            Ok(path.to_path_buf())
        } else {
            Ok(self.base_path.join(path))
        }
    }

    /// 一括操作
    pub fn batch_operation<F>(
        &self,
        paths: &[PathBuf],
        operation: F,
    ) -> Result<Vec<Result<(), SiggError>>, SiggError>
    where
        F: Fn(&Self, &Path) -> Result<(), SiggError>,
    {
        let results: Vec<_> = paths
            .iter()
            .map(|path| operation(self, path))
            .collect();
        
        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: PathBuf,
    pub size: u64,
    pub is_file: bool,
    pub is_dir: bool,
    pub modified: Option<std::time::SystemTime>,
}

/// ファイルウォッチャー（変更監視）
pub struct FileWatcher {
    watched_paths: Vec<(PathBuf, u64)>, // (path, last_modified_timestamp)
}

impl FileWatcher {
    pub fn new() -> Self {
        Self {
            watched_paths: Vec::new(),
        }
    }

    pub fn watch(&mut self, path: PathBuf) -> Result<(), SiggError> {
        let metadata = fs::metadata(&path)
            .map_err(|e| SiggError::runtime(format!("メタデータ取得エラー: {}", e)))?;
        
        let modified = metadata
            .modified()
            .map_err(|e| SiggError::runtime(format!("更新時刻取得エラー: {}", e)))?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        self.watched_paths.push((path, modified));
        Ok(())
    }

    pub fn check_changes(&mut self) -> Result<Vec<PathBuf>, SiggError> {
        let mut changed = Vec::new();
        
        for (path, last_modified) in &mut self.watched_paths {
            if let Ok(metadata) = fs::metadata(&mut *path) {
                if let Ok(modified) = metadata.modified() {
                    let current = modified
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    
                    if current > *last_modified {
                        changed.push(path.clone());
                        *last_modified = current;
                    }
                }
            }
        }
        
        Ok(changed)
    }
}