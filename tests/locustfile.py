from locust import HttpUser, task


class ApiUser(HttpUser):
    @task
    def get_plp(self):
        self.client.post("/search/", headers={'Authorization': 'Bearer eyJhbGciOi'})
