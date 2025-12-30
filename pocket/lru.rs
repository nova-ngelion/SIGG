use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

pub struct LruCache<K, V> {
    cap: usize,
    tick: u64,
    map: HashMap<K, (V, u64)>,         // value, gen
    q: VecDeque<(K, u64)>,             // key, gen at access time
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            tick: 0,
            map: HashMap::new(),
            q: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize { self.map.len() }
    pub fn cap(&self) -> usize { self.cap }

    pub fn get(&mut self, k: &K) -> Option<&V> {
        if self.map.contains_key(k) {
            self.tick += 1;
            // gen更新
            let gen = {
                let (_, g) = self.map.get_mut(k).unwrap();
                *g = self.tick;
                *g
            };
            self.q.push_back((k.clone(), gen));
            self.evict_if_needed();
            return self.map.get(k).map(|(v, _)| v);
        }
        None
    }

    pub fn insert(&mut self, k: K, v: V) {
        self.tick += 1;
        let gen = self.tick;
        self.map.insert(k.clone(), (v, gen));
        self.q.push_back((k, gen));
        self.evict_if_needed();
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.map.contains_key(k)
    }

    pub fn remove(&mut self, k: &K) {
        self.map.remove(k);
    }

    fn evict_if_needed(&mut self) {
        while self.map.len() > self.cap {
            if let Some((k, gen)) = self.q.pop_front() {
                // キューの gen が「そのキーの最新 gen」と一致してたら本当にLRUなので削除
                let remove = self.map.get(&k).map(|(_, g)| *g == gen).unwrap_or(false);
                if remove {
                    self.map.remove(&k);
                }
            } else {
                break;
            }
        }
        // キューが膨らみ続けないように軽く掃除（任意）
        if self.q.len() > self.cap * 8 {
            self.compact();
        }
    }

    fn compact(&mut self) {
        let mut newq = VecDeque::with_capacity(self.cap * 2);
        // 末尾側（新しいアクセス）から拾って最新genだけ残す
        for (k, gen) in self.q.iter().rev() {
            let keep = self.map.get(k).map(|(_, g)| *g == *gen).unwrap_or(false);
            if keep {
                newq.push_front((k.clone(), *gen));
                if newq.len() >= self.cap * 2 { break; }
            }
        }
        self.q = newq;
    }
}
