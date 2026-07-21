using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;

public class CSVPlayer : MonoBehaviour
{
    public Transform body1;
    public Transform body2;
    public Transform body3;

    public float frameRate;
    public int framesCount;

    public List<Vector2[]> frames = new();

    public bool playPositions;
    
    public int currentFrame;

    private void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, "predicted_positions.csv");
        string[] lines = File.ReadAllLines(path);

        for (int i = 1; i < lines.Length; i++)
        {
            string[] c = lines[i].Split(",");

            Vector2[] frame = new Vector2[3];

            frame[0] = new Vector2(float.Parse(c[0]), float.Parse(c[1]));
            frame[1] = new Vector2(float.Parse(c[4]), float.Parse(c[5]));
            frame[2] = new Vector2(float.Parse(c[8]), float.Parse(c[9]));
            
            frames.Add(frame);
        }
        
    }

    float timer;

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= 1f / frameRate)
        {
            timer -= 1f / frameRate;
            NextFrame();
        }
    }
    
    void NextFrame()
    {
        if (currentFrame >= framesCount) currentFrame -= framesCount;

        if (currentFrame >= 0)
        {
            body1.position = frames[currentFrame][0];
            body2.position = frames[currentFrame][1];
            body3.position = frames[currentFrame][2];
        }
        
        if (playPositions) currentFrame++;
    }
}
