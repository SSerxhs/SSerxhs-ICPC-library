#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
struct Vector
{
	int *a;
	int n,m;
};
typedef struct Vector vector;
void swap(int *x,int *y) { int z=*x; *x=*y; *y=z; }
void reverse(int *l,int *r) { while (l<r) swap(l++,--r); }
int *lower_bound(int *l,int *r,int x)
{
	int n=r-l;
	while (n>0)
	{
		int *mid=l+(n>>1);
		if (*mid<x) l=mid+1,n-=n>>1;
		else n>>=1;
	}
	return l;
}
int *upper_bound(int *l,int *r,int x)
{
	int n=r-l;
	while (n>0)
	{
		int *mid=l+(n>>1);
		if (!(x<*mid)) l=mid+1,n-=n>>1;
		else n>>=1;
	}
	return l;
}
int cmp(const void *a,const void *b) { return (*(int *)a-*(int *)b); }
void sort(int *l,int *r) { qsort(l,r-l,sizeof(int),cmp); }
int read()
{
	int c=getchar(),fh=1;
	while ((c<48)||(c>57))
	{
		if (c=='-') { c=getchar(); fh=-1; break; }
		c=getchar();
	}
	int x=c^48; c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+(c^48);
		c=getchar();
	}
	return x*fh;
}
void readline(int *a,int n)
{
	for (int i=0; i<n; i++) a[i]=read();
}
void write(int *a,int n)
{
	for (int i=0; i<n; i++) printf("%d%c",a[i]," \n"[i==n-1]);
}
int main()
{
}