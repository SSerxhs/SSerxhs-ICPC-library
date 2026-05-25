#include<cstdio>
#include<algorithm>
const int N = 524288, md = 998244353, g3 = (md + 1) / 3;
typedef long long LL;
int n, m, A[N], B[N], fac[N], iv[N], rev[N], C[N], g[20][N], lim, M;
int pow(int a, int b) {
    int ret = 1;
    for (; b; b >>= 1, a = (LL)a * a % md)if (b & 1)ret = (LL)ret * a % md;
    return ret;
}
void upd(int &a) { a += a >> 31 & md; }
void init(int n) {
    int l = -1;
    for (lim = 1; lim < n; lim <<= 1)++l; M = l + 1;
    for (int i = 1; i < lim; ++i)
        rev[i] = ((rev[i >> 1]) >> 1) | ((i & 1) << l);
}
void NTT(int *a, int f) {
    for (int i = 1; i < lim; ++i)if (i < rev[i])std::swap(a[i], a[rev[i]]);
    for (int i = 0; i < M; ++i) {
        const int *G = g[i], c = 1 << i;
        for (int j = 0; j < lim; j += c << 1)
            for (int k = 0; k < c; ++k) {
                const int x = a[j + k], y = a[j + k + c] * (LL)G[k] % md;
                upd(a[j + k] += y - md), upd(a[j + k + c] = x - y);
            }
    }
    if (!f) {
        const int iv = pow(lim, md - 2);
        for (int i = 0; i < lim; ++i)a[i] = (LL)a[i] * iv % md;
        std::reverse(a + 1, a + lim);
    }
}
int main() {
    scanf("%d%d", &n, &m); ++n, ++m;
    for (int i = 0; i < 20; ++i) {
        int *G = g[i];
        G[0] = 1;
        const int gi = G[1] = pow(3, (md - 1) / (1 << i + 1));
        for (int j = 2; j < 1 << i; ++j)G[j] = (LL)G[j - 1] * gi % md;
    }
    for (int i = 0; i < n; ++i)scanf("%d", A + i);
    for (int i = 0; i < m; ++i)scanf("%d", B + i);
    for (int i = *fac = 1; i < N; ++i)
        fac[i] = fac[i - 1] * (LL)i % md;
    iv[N - 1] = pow(fac[N - 1], md - 2);
    for (int i = N - 2; ~i; --i)iv[i] = (i + 1LL) * iv[i + 1] % md;
    init(n + m << 1);
    for (int i = 0; i < n + m - 1; ++i)C[i] = iv[i];
    NTT(A, 1), NTT(B, 1), NTT(C, 1);
    for (int i = 0; i < lim; ++i)A[i] = (LL)A[i] * C[i] % md, B[i] = (LL)B[i] * C[i] % md;
    NTT(A, 0), NTT(B, 0);
    for (int i = 0; i < lim; ++i)C[i] = 0;
    for (int i = 0; i < n + m - 1; ++i)
        C[i] = (i & 1) ? md - iv[i] : iv[i];
    for (int i = 0; i < lim; ++i)A[i] = (LL)A[i] * B[i] % md * fac[i] % md;
    for (int i = n + m - 1; i < lim; ++i)A[i] = 0;
    NTT(A, 1), NTT(C, 1);
    for (int i = 0; i < lim; ++i)A[i] = (LL)A[i] * C[i] % md;
    NTT(A, 0);
    for (int i = 0; i < n + m - 1; ++i)printf("%d%c", A[i], " \n"[i == n + m - 2]);
    return 0;
}
