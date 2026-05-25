#include"bits/stdc++.h"
using ll = long long;
using lll = __int128;
using namespace std;

void myw(lll x) {
	if (!x) return;
	myw(x / 10); printf("%d", (int)(x % 10));
}

struct vec {
	ll x, y;
	vec(ll x0 = 0, ll y0 = 0) { x = x0, y = y0; }
	vec operator +(const vec b) { return vec(x + b.x, y + b.y); }
};

ll N;
vec stk[1000005]; int len;
vec P;
vec L, R;

bool ninR(vec a) { return N < (lll)a.x * a.y; }
bool steep(ll x, vec a) { return (lll)N * a.x <= (lll)x * x * a.y; }

lll Solve() {
	len = 0;
	ll cbr = cbrt(N), sqr = sqrt(N);
	P.x = N / sqr, P.y = sqr + 1;
	lll ans = 0;
	stk[++len] = vec(1, 0); stk[++len] = vec(1, 1);
	while (1) {
		L = stk[len--];
		while (ninR(vec(P.x + L.x, P.y - L.y)))
			ans += (lll)P.x * L.y + (lll)(L.y + 1) * (L.x - 1) / 2,
			P.x += L.x, P.y -= L.y;
		if (P.y <= cbr) break;
		R = stk[len];
		while (!ninR(vec(P.x + R.x, P.y - R.y))) L = R, R = stk[--len];
		while (1) {
			vec mid = L + R;
			if (ninR(vec(P.x + mid.x, P.y - mid.y))) R = stk[++len] = mid;
			else if (steep(P.x + mid.x, R)) break;
			else L = mid;
		}
	}
	for (int i = 1; i < P.y; i++) ans += N / i;
	return ans * 2 - sqr * sqr;
}

int T;

int main() {
	scanf("%d", &T);
	while (T--) {
		scanf("%lld", &N);
		myw(Solve()); printf("\n");
	}
}
