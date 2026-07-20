using System.IO;
using System.Text;
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
    public int episodeLength;
    public int numEpisodes;
    public int episode;

    public int frame;

    private StreamWriter writer;

    private void Start()
    {
        Random.InitState((int)System.DateTime.Now.Ticks);
        positions = new Vector2[nBodies];
        velocities = new Vector2[nBodies];
        gameObjects = new GameObject[nBodies];
        
        for (int i = 0; i < nBodies; i++)
        {
            positions[i] = Random.insideUnitCircle * spawnRadius;
            velocities[i] = Vector2.zero;
            gameObjects[i] = Instantiate(prefab, positions[i], Quaternion.identity);
        }

        string path = Path.Combine(Application.dataPath, "../simulation_data.csv");
        writer = new StreamWriter(path, append: false, encoding: Encoding.UTF8);

        var header = new StringBuilder();
        for (int i = 0; i < nBodies; i++)
        {
            header.Append($"b{i}x, b{i}y, b{i}vx, b{i}vy");
            if (i < nBodies - 1) header.Append(",");
        }
        writer.WriteLine(header.ToString());
    }

    private void Update()
    {
        if (episode >= numEpisodes)
        {
            if (writer != null)
            {
                writer.Close();
                writer = null;
                print("CSV Saved");
            }
            return;
        }
        
        frame++;
        
        if (frame % episodeLength == 1 && frame > 1)
        {
            ResetSimulation();
        }

        // Physics update
        for (int i = 0; i < nBodies; i++)
        {
            for (int j = 0; j < nBodies; j++)
            {
                if (i == j) continue;
                Vector2 direction      = positions[j] - positions[i];
                float distanceSquared  = math.dot(direction, direction) + softening * softening;
                float distance         = math.sqrt(distanceSquared);
                float forceMagnitude   = mass / distanceSquared;
                velocities[i]         += forceMagnitude * (direction / distance) * 0.01f;
            }
            positions[i] += velocities[i] * 0.01f;
        }

        
        UpdateGameObjects();

        if (writer != null)
        {
            var row = new StringBuilder();
            for (int i = 0; i < nBodies; i++)
            {
                row.Append($"{positions[i].x:F6},{positions[i].y:F6},{velocities[i].x:F6},{velocities[i].y:F6}");
                if (i < nBodies - 1) row.Append(",");
            }
            writer.WriteLine(row.ToString());
        }
    }
    
    private void ResetSimulation()
    {
        episode++;
        
        print("Episode: " + episode);
            
        for (int i = 0; i < nBodies; i++)
        {
            positions[i] = Random.insideUnitCircle * spawnRadius;
            velocities[i] = new Vector2(Random.Range(-1f, 1f), Random.Range(-1f, 1f));
            gameObjects[i].transform.position = new Vector3(positions[i].x, positions[i].y, 0.0f);
        }
    }

    private void UpdateGameObjects()
    {
        for (int i = 0; i < nBodies; i++)
        {
            gameObjects[i].transform.position = new Vector3(positions[i].x, positions[i].y, 0.0f);
        }
    }
}
