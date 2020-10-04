#include<bits/stdc++.h>
using namespace std;
#define inp(k)           ll k;cin>>k;
#define in2(n, k)       ll n, k; cin>>n;cin>>k;
#define ll long long
#define dl double
#define ch char
#define f(i,a,b) for(ll i=a;i<b;i++)
#define rf(i,a,b) for(ll i=a-1;i>=b;i--)
#define Godspeed                ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);
#define vpl vector< pair<ll,ll> >
#define vpi vector< pair<int,int> >
#define vi vector<int>
#define vl vector<ll>
#define vdl vector<dl>
#define vch vector<char>
#define pll pair<ll,ll>
#define pb push_back
#define mp make_pair
#define fr first
#define se second
#define PI 3.14159265
bool comp(pair<long long int ,long long int > &a,pair<long long int ,long long int > &b)
{
	return a.second > b.second;
}
#include <limits.h> 
#include <stdio.h> 

#define V 9 

int minDistance(int dist[], bool sptSet[]) 
{ 
	int min = INT_MAX, min_index; 

	for (int v = 0; v < V; v++) 
		if (sptSet[v] == false && dist[v] <= min) 
			min = dist[v], min_index = v; 

	return min_index; 
} 
//main jadoogar hai mera naam goga

void printSolution(int dist[]) 
{ 
	printf("Vertex \t\t Distance from Source\n"); 
	for (int i = 0; i < V; i++) 
		printf("%d \t\t %d\n", i, dist[i]); 
} 
//mujh jaisa nahin koi hoga
void dijkstra(int graph[V][V], int src) 
{ 
	int dist[V]; 

	bool sptSet[V];

	for (int i = 0; i < V; i++) 
		dist[i] = INT_MAX, sptSet[i] = false; dist[src] = 0; 

	for (int count = 0; count < V - 1; count++) { 
		int u = minDistance(dist, sptSet); sptSet[u] = true; 
		for (int v = 0; v < V; v++) 

			if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX 
				&& dist[u] + graph[u][v] < dist[v]) dist[v] = dist[u] + graph[u][v]; 
	} 

	printSolution(dist); 
} 
//main jadoogar hai mera naam goga
//mujh jaisa nahin koi hoga

int main() 
{ 
	int graph[V][V] = { { 0, 4, 0, 0, 0, 0, 0, 8, 0 }, 
						{ 4, 0, 8, 0, 0, 0, 0, 11, 0 }, 
						{ 0, 8, 0, 7, 0, 4, 0, 0, 2 }, 
						{ 0, 0, 7, 0, 9, 14, 0, 0, 0 }, 
						{ 0, 0, 0, 9, 0, 10, 0, 0, 0 }, 
						{ 0, 0, 4, 14, 10, 0, 2, 0, 0 }, 
						{ 0, 0, 0, 0, 0, 2, 0, 1, 6 }, 
						{ 8, 11, 0, 0, 0, 0, 1, 0, 7 }, 
						{ 0, 0, 2, 0, 0, 0, 6, 7, 0 } }; 

	dijkstra(graph, 0); 

	return 0; 
} 
