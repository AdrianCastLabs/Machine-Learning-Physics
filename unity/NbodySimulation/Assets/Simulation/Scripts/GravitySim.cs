using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

public class GravitySim : MonoBehaviour
{
    public GameObject prefab;
    
    public Vector2[] positions;
    public Vector2[] velocities;
    public GameObject[] gameObjects;

    public float spawnRadius;
    public float softening;
    public float mass;
    public int nBodies;

    public int frame;

    private void Start()
    {
        positions = new Vector2[nBodies];
        velocities = new Vector2[nBodies];
        gameObjects = new GameObject[nBodies];
        
        for (int i = 0; i < nBodies; i++)
        {
            positions[i] = Random.insideUnitCircle * spawnRadius;
            velocities[i] = Vector2.zero;
            gameObjects[i] = Instantiate(prefab, positions[i], Quaternion.identity);
        }
    }

    private void Update()
    {
        frame += 1;

        if (frame % 500 == 1)
        {
            print("Simulation reset");
            
            for (int i = 0; i < nBodies; i++)
            {
                positions[i] = Random.insideUnitCircle * spawnRadius;
                velocities[i] = Random.insideUnitCircle;
                gameObjects[i].transform.position = new Vector3(positions[i].x, positions[i].y, 0.0f);
            }
        }
        
        for (int i = 0; i < nBodies; i++)
        {
            for (int j = 0; j < nBodies; j++)
            {
                if (i == j) continue;
                Vector2 otherPosition = positions[j];
                Vector2 direction = otherPosition - positions[i];
                float distanceSquared = math.dot(direction, direction);
                distanceSquared += softening * softening;
                float distance = math.sqrt(distanceSquared);
                float forceMagnitude = mass / distanceSquared;
                direction /= distance;
                velocities[i] += forceMagnitude * direction;

            }

            positions[i] += velocities[i];

        }
        
        UpdateGameObjects();
    }

    private void UpdateGameObjects()
    {
        for (int i = 0; i < nBodies; i++)
        {
            gameObjects[i].transform.position = new Vector3(positions[i].x, positions[i].y, 0.0f);
        }
    }
}
