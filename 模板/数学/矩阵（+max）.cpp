template<typename T,int n> struct matrix
{
#define all(x) (x).begin(),(x).end()
	array<array<T,n>,n> a;
	matrix(char c='O')
	{
		int i;
		if (c=='O') for (i=0; i<n; i++) fill(all(a[i]),-inf);
		else if (c=='E') for (i=0; i<n; i++) fill(all(a[i]),-inf),a[i][i]=0;
		else assert(0);
	}
	matrix(char c,int x)
	{

	}
	matrix operator+(const matrix &o) const
	{
		matrix r;
		int i,j,k;
		for (k=0; k<n; k++) for (i=0; i<n; i++) for (j=0; j<n; j++) r.a[i][j]=max(r.a[i][j],a[i][k]+o.a[k][j]);
		return r;
	}
};
template<int n> ostream &operator<<(ostream &cout,const matrix<n> &o)
{
	int i,j;
	for (i=0; i<n; i++) for (j=0; j<n; j++) cout<<o.a[i][j]<<" \n"[j+1==n];
	return cout;
}