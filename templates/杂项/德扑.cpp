struct Q
{
	int suit, rank;
	bool operator<(const Q &o) const { return pair{rank, suit} < pair{o.rank, o.suit}; }
	bool operator==(const Q &o) const { return pair{rank, suit} == pair{o.rank, o.suit}; }
};
auto solve = [&](vector<Q> a) {
	vector<int> res;
	vector<int> cnt(15);
	for (auto [s, r] : a) ++cnt[r];
	sort(all(a));
	int i;
	bool is_flush = 1, is_str = 0;
	for (i = 1; i < 5; i++) is_flush &= a[i].suit == a[0].suit;
	is_str = *max_element(all(cnt)) == 1 && a[0].rank + 4 == a[4].rank;
	vector<int> b(6);
	for (i = 1; i < 6; i++) b[i] = a[i - 1].rank;
	sort(1 + all(b), [&](int x, int y) {
		return pair{cnt[x], x} > pair{cnt[y], y};
	});
	if (b == vector{0, 12, 3, 2, 1, 0}) is_str = 1, b[1] = 0;
	if (is_flush && is_str) return b[0] = 9, b;
	if (cnt[b[1]] == 4) return b[0] = 8, b;
	if (cnt[b[1]] == 3 && cnt[b[4]] == 2) return b[0] = 7, b;
	if (is_flush) return b[0] = 6, b;
	if (is_str) return b[0] = 5, b;
	if (cnt[b[1]] == 3) return b[0] = 4, b;
	if (cnt[b[1]] == 2 && cnt[b[3]] == 2) return b[0] = 3, b;
	if (cnt[b[1]] == 2) return b[0] = 2, b;
	return b;
};
auto turn = [&](string s) {
	Q res = Q{"SHDC"s.find(s[0]), "23456789TJQKA"s.find(s[1])};
	return res;
};

